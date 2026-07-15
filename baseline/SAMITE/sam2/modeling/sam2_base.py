# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Modified by the SENTRY Authors in 2026 for SENTRY integration.

import os
import sys
from loguru import logger

import numpy as np
import cv2

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames
from sam2.utils.kalman_filter import KalmanFilter

from einops import rearrange

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.dtype_override = dtype_override
        self.channels = channels

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels, dtype_override)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels
    

class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        # Whether to use SAMURAI
        samurai_mode: bool = False,
        # Hyperparameters for SAMURAI
        stable_frames_threshold: int = 15,
        stable_ious_threshold: float = 0.3,
        min_obj_score_logits: float = -1,
        kf_score_weight: float = 0.15,
        memory_bank_iou_threshold: float = 0.5,
        memory_bank_obj_score_threshold: float = 0.0,
        memory_bank_kf_score_threshold: float = 0.0,
        # Whether to use SAMITE
        samite_mode : bool = False,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Whether to use original SAM 2, SAMURAI or SAMITE
        self.samurai_mode = samurai_mode
        self.samite_mode = samite_mode

        # Init Kalman Filter
        self.kf = KalmanFilter()
        self.kf_mean = None
        self.kf_covariance = None
        self.stable_frames = 0
        self.use_second_prior_iou_thr = 0.8
        
        # Debug purpose
        self.history = {} # debug
        self.frame_cnt = 0 # debug

        # Hyperparameters for SAMURAI
        self.stable_frames_threshold = stable_frames_threshold
        self.stable_ious_threshold = stable_ious_threshold
        self.min_obj_score_logits = min_obj_score_logits
        self.kf_score_weight = kf_score_weight
        self.memory_bank_iou_threshold = memory_bank_iou_threshold
        self.memory_bank_obj_score_threshold = memory_bank_obj_score_threshold
        self.memory_bank_kf_score_threshold = memory_bank_kf_score_threshold
        
        # ========================================
        # Positional Encoding
        # ========================================
        self.pe = PositionalEncodingPermute2D(channels=256)

        if not (self.samurai_mode or self.samite_mode):
            print(f"\033[93mSAM 2 mode.\033[0m")
        elif self.samurai_mode:
            print(f"\033[93mSAMURAI mode.\033[0m")
        elif self.samite_mode:
            print(f"\033[93mSAMITE mode.\033[0m")

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def check_results(self, curr_feat, curr_mask, prev_feat, prev_mask, args):
        B = curr_feat.size(0)
        
        # current feats and preds
        curr_feat = curr_feat           # 3, c, h, w
        curr_mask = F.interpolate(curr_mask, size=prev_feat.size()[-2:], mode='bilinear')
        curr_mask = F.sigmoid(curr_mask)
        curr_mask[curr_mask >= 0.5] = 1
        curr_mask[curr_mask < 0.5] = 0  # 3, 1, h, w
        
        # previous feats and preds
        prev_feat = prev_feat.squeeze(0)                # n, c, h, w
        prev_mask = prev_mask.squeeze(0).unsqueeze(1)   # n, 1, h, w
        prev_num = prev_feat.size(0)
        
        # current pros
        curr_pro_fg = self.memory_attention.Weighted_GAP(curr_feat, curr_mask)       # 3, c, 1
        curr_pro_bg = self.memory_attention.Weighted_GAP(curr_feat, 1. - curr_mask)  # 3, c, 1
        
        curr_pro_fg = rearrange(curr_pro_fg, 'b c 1 n -> 1 c b n')  # 1, c, 3, 1
        curr_pro_bg = rearrange(curr_pro_bg, 'b c 1 n -> 1 c b n')  # 1, c, 3, 1
        
        curr_pro_fg = curr_pro_fg.expand(prev_num, -1, -1, -1)      # n, c, 3, 1
        curr_pro_bg = curr_pro_bg.expand(prev_num, -1, -1, -1)      # n, c, 3, 1
    
        # positional prior
        enc = self.pe(curr_feat)    # 3, c, h, w
        enc_priors = self.memory_attention.generate_prior_enc(enc, enc, [curr_mask], fts_size=prev_feat.size()[-2:])
        enc_priors = enc_priors[0]  # 3, 1, h, w
        enc_priors = rearrange(enc_priors, 'n 1 h w -> 1 n h w')  # 1, 3, h, w
        enc_priors = enc_priors.expand(prev_num, -1, -1, -1)      # n, 3, h, w
        
        # previous priors
        _, _, _, prev_prior = self.memory_attention.generate_prior_feat_enc_batch(prev_feat, curr_pro_fg, curr_pro_bg, enc_priors, fts_size=prev_feat.size()[-2:])
        thr = 0.5
        prev_prior[prev_prior >= thr] = 1
        prev_prior[prev_prior < thr] = 0
        
        # measure iou score between prior and gt
        prev_prior = prev_prior.to(torch.float32)  # n, 3, h, w
        prev_mask = prev_mask.to(torch.float32)    # n, 1, h, w
        
        best_ind = 0
        best_iou = -1
        ious = []
        for idx in range(B):
            intersection, union, _ = self.calculate_IoU(prev_prior[:, idx, ...].contiguous(), prev_mask[:, 0, ...].contiguous(), K=2)
            iou = intersection[1].cpu().numpy() / (union[1].cpu().numpy() + 1e-7)
            ious.append(iou)
            if iou >= best_iou:
                best_iou = iou
                best_ind = idx

        if best_iou < 0.7:  # beta
            best_ind = 0
                
        return best_iou, best_ind

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        empty_first=False,  # when batch size is more than 1, whether embed first mask or not
        prev_feat=None,
        prev_mask=None,
        args=None,
        prior=None,
        update_kalman=True
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
            empty_first=empty_first  # when batch size is more than 1, whether embed first mask or not
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > self.min_obj_score_logits

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        kf_ious = None
        batch_inds = torch.arange(B, device=device)
        if multimask_output and (self.samurai_mode or self.samite_mode):  # SAMURAI
            if self.kf_mean is None and self.kf_covariance is None or self.stable_frames == 0:
                best_iou_inds = torch.argmax(ious, dim=-1)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)  # b, 1, h, w
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]

                # ========================================
                # select best mask along batch dimension
                # ========================================
                if empty_first and prev_feat is not None and prev_mask is not None:
                    if B in [2, 3]:
                        best_iou, best_ind = self.check_results(
                            curr_feat=backbone_features,
                            curr_mask=high_res_masks,
                            prev_feat=prev_feat,
                            prev_mask=prev_mask,
                            args=args,
                        )
                        
                        ious = ious[best_ind: best_ind + 1, ...]
                        low_res_multimasks = low_res_multimasks[best_ind: best_ind + 1, ...]
                        high_res_multimasks = high_res_multimasks[best_ind: best_ind + 1, ...]
                        low_res_masks = low_res_masks[best_ind: best_ind + 1, ...]
                        high_res_masks = high_res_masks[best_ind: best_ind + 1, ...]
                        sam_output_token = sam_output_token[best_ind: best_ind + 1, ...]
                        sam_output_tokens = sam_output_tokens[best_ind: best_ind + 1, ...]
                        object_score_logits = object_score_logits[best_ind: best_ind + 1, ...]
                        best_iou_inds = best_iou_inds[best_ind: best_ind + 1, ...]
                        if self.pred_obj_scores:
                            is_obj_appearing = is_obj_appearing[best_ind: best_ind + 1, ...]
                    else:
                        print("Currently only support bs 2 or 3.")
                        sys.exit(0)
                
                if update_kalman:
                    non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)  # use sample 0 to update kalman statistics
                    if len(non_zero_indices) == 0:
                        high_res_bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(dim=0).values
                        y_max, x_max = non_zero_indices.max(dim=0).values
                        high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                    self.kf_mean, self.kf_covariance = self.kf.initiate(self.kf.xyxy_to_xyah(high_res_bbox))
                    self.frame_cnt += 1
                    self.stable_frames += 1
            elif self.stable_frames < self.stable_frames_threshold:
                if update_kalman:
                    self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
                best_iou_inds = torch.argmax(ious, dim=-1)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                   
                # ========================================
                # select best mask along batch dimension
                # ========================================
                if empty_first and prev_feat is not None and prev_mask is not None:
                    if B in [2, 3]:
                        best_iou, best_ind = self.check_results(
                            curr_feat=backbone_features,
                            curr_mask=low_res_masks,
                            prev_feat=prev_feat,
                            prev_mask=prev_mask,
                            args=args,
                        )
                        
                        ious = ious[best_ind: best_ind + 1, ...]
                        low_res_multimasks = low_res_multimasks[best_ind: best_ind + 1, ...]
                        high_res_multimasks = high_res_multimasks[best_ind: best_ind + 1, ...]
                        low_res_masks = low_res_masks[best_ind: best_ind + 1, ...]
                        high_res_masks = high_res_masks[best_ind: best_ind + 1, ...]
                        sam_output_token = sam_output_token[best_ind: best_ind + 1, ...]
                        sam_output_tokens = sam_output_tokens[best_ind: best_ind + 1, ...]
                        object_score_logits = object_score_logits[best_ind: best_ind + 1, ...]
                        best_iou_inds = best_iou_inds[best_ind: best_ind + 1, ...]
                        if self.pred_obj_scores:
                            is_obj_appearing = is_obj_appearing[best_ind: best_ind + 1, ...]
                    else:
                        print("Currently only support bs 2 or 3.")
                        sys.exit(0)
                
                if update_kalman:
                    non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                    if len(non_zero_indices) == 0:
                        high_res_bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(dim=0).values
                        y_max, x_max = non_zero_indices.max(dim=0).values
                        high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                    if ious[0][best_iou_inds[0]] > self.stable_ious_threshold:
                        self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_bbox))
                        self.stable_frames += 1
                    else:
                        self.stable_frames = 0
                    self.frame_cnt += 1
            else:
                if update_kalman:
                    self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
                high_res_multibboxes = []
                for i in range(ious.shape[1]):
                    non_zero_indices = torch.argwhere(high_res_multimasks[batch_inds, i].unsqueeze(1)[0][0] > 0.0)
                    if len(non_zero_indices) == 0:
                        high_res_multibboxes.append([0, 0, 0, 0])
                    else:
                        y_min, x_min = non_zero_indices.min(dim=0).values
                        y_max, x_max = non_zero_indices.max(dim=0).values
                        high_res_multibboxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])
                # compute the IoU between the predicted bbox and the high_res_multibboxes
                kf_ious = torch.tensor(self.kf.compute_iou(self.kf_mean[:4], high_res_multibboxes), device=device)
                # weighted iou
                weighted_ious = self.kf_score_weight * kf_ious + (1 - self.kf_score_weight) * ious
                if B > 1:  # do not have kalman filter for frame 1
                    weighted_ious[1:, ...] = ious[1:, ...]
                best_iou_inds = torch.argmax(weighted_ious, dim=-1)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                
                # ========================================
                # select best mask along batch dimension
                # ========================================
                if empty_first and prev_feat is not None and prev_mask is not None:
                    if B in [2, 3]:
                        best_iou, best_ind = self.check_results(
                            curr_feat=backbone_features,
                            curr_mask=low_res_masks,
                            prev_feat=prev_feat,
                            prev_mask=prev_mask,
                            args=args,
                        )
                        
                        ious = ious[best_ind: best_ind + 1, ...]
                        low_res_multimasks = low_res_multimasks[best_ind: best_ind + 1, ...]
                        high_res_multimasks = high_res_multimasks[best_ind: best_ind + 1, ...]
                        low_res_masks = low_res_masks[best_ind: best_ind + 1, ...]
                        high_res_masks = high_res_masks[best_ind: best_ind + 1, ...]
                        sam_output_token = sam_output_token[best_ind: best_ind + 1, ...]
                        sam_output_tokens = sam_output_tokens[best_ind: best_ind + 1, ...]
                        object_score_logits = object_score_logits[best_ind: best_ind + 1, ...]
                        best_iou_inds = best_iou_inds[best_ind: best_ind + 1, ...]
                        if self.pred_obj_scores:
                            is_obj_appearing = is_obj_appearing[best_ind: best_ind + 1, ...]
                    else:
                        print("Currently only support bs 2 or 3.")
                        sys.exit(0)
                
                if update_kalman:
                    self.frame_cnt += 1

                    if ious[0][best_iou_inds[0]] < self.stable_ious_threshold:
                        self.stable_frames = 0
                    else:
                        self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_multibboxes[best_iou_inds[0]]))
            
        elif multimask_output and not (self.samurai_mode or self.samite_mode):  # original SAM 2 / SAMITE
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                
            # ========================================
            # select best mask along batch dimension
            # ========================================
            if empty_first and prev_feat is not None and prev_mask is not None:
                if B in [2, 3]:
                    best_iou, best_ind = self.check_results(
                        curr_feat=backbone_features,
                        curr_mask=low_res_masks,
                        prev_feat=prev_feat,
                        prev_mask=prev_mask,
                        args=args,
                    )
                    
                    ious = ious[best_ind: best_ind + 1, ...]
                    low_res_multimasks = low_res_multimasks[best_ind: best_ind + 1, ...]
                    high_res_multimasks = high_res_multimasks[best_ind: best_ind + 1, ...]
                    low_res_masks = low_res_masks[best_ind: best_ind + 1, ...]
                    high_res_masks = high_res_masks[best_ind: best_ind + 1, ...]
                    sam_output_token = sam_output_token[best_ind: best_ind + 1, ...]
                    sam_output_tokens = sam_output_tokens[best_ind: best_ind + 1, ...]
                    object_score_logits = object_score_logits[best_ind: best_ind + 1, ...]
                    best_iou_inds = best_iou_inds[best_ind: best_ind + 1, ...]
                    if self.pred_obj_scores:
                        is_obj_appearing = is_obj_appearing[best_ind: best_ind + 1, ...]
                else:
                    print("Currently only support bs 2 or 3.")
                    sys.exit(0)    
        else:
            best_iou_inds = torch.tensor(0).to(device).view(1)
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
            if (self.samurai_mode or self.samite_mode):
                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                self.kf_mean, self.kf_covariance = self.kf.initiate(self.kf.xyxy_to_xyah(high_res_bbox))
                
        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            ious[0, best_iou_inds[0]],
            kf_ious[best_iou_inds[0]] if kf_ious is not None else None,
            prior[0] if prior is not None else None
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _, _, _, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
        
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        args=None
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        pro_fgs, pro_bgs, masks = [], [], []
        prior = None
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_prev_feat = []
            to_cat_memory, to_cat_memory_pos_embed, to_cat_memory_mask = [], [], []
            to_cat_pro_fg, to_cat_pro_bg = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]  # always preserve the first frame
            valid_indices_complete = []
            
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval

            if self.samurai_mode or self.samite_mode:
                valid_indices = []
                feats = []
                max_num_indices = args.max_num_indices if args is not None else 30
                if frame_idx > 1:
                    # ========================================
                    # Select valid previous frames as candidates
                    # - iou score is valid;
                    # - object appearing score is valid;
                    # - kalman motion score is valid.
                    # ========================================
                    for i in range(frame_idx - 1, 1, -1):  # Iterate backwards through previous frames
                        iou_score = output_dict["non_cond_frame_outputs"][i]["best_iou_score"]  # Get mask affinity score
                        obj_score = output_dict["non_cond_frame_outputs"][i]["object_score_logits"]  # Get object score
                        kf_score = output_dict["non_cond_frame_outputs"][i]["kf_score"] if "kf_score" in output_dict["non_cond_frame_outputs"][i] else None  # Get motion score if available
                        # Check if the scores meet the criteria for being a valid index
                        if iou_score.mean() > self.memory_bank_iou_threshold and \
                        obj_score.mean() > self.memory_bank_obj_score_threshold and \
                        (kf_score is None or kf_score.item() > self.memory_bank_kf_score_threshold):
                            # indices
                            valid_indices_complete.insert(0, i)  # all valid indices that satisfies kalman threshold
                            valid_indices.insert(0, i)  # compared to inds, valid_indices will be filtered
                            
                            # features
                            feats.insert(0, output_dict["non_cond_frame_outputs"][i]["image_features"].to(device, non_blocking=True).unsqueeze(1))  # b, 1, c, h, w
                            
                            # prototypes
                            pro_fgs.insert(0, output_dict["non_cond_frame_outputs"][i]["pro_fg"])
                            pro_bgs.insert(0, output_dict["non_cond_frame_outputs"][i]["pro_bg"])
                            
                            # masks
                            mask = output_dict["non_cond_frame_outputs"][i]["pred_masks"].to(device, non_blocking=True)
                            mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)  # (b=1, c=1, h=32, w=32)
                            mask = F.sigmoid(mask)
                            mask[mask >= 0.5] = 1
                            mask[mask < 0.5] = 0
                            masks.insert(0, mask)
                            
                        if len(valid_indices) >= max_num_indices:
                            break
                            
                    # ========================================
                    # Select good candidates
                    # ========================================
                    num_maskmem = self.num_maskmem - len(t_pos_and_prevs)  # already has 1 cond frame (frame #0, with gt bbox as input)
                    if len(pro_fgs) > num_maskmem:
                        pro_fg = torch.cat(pro_fgs, dim=2).squeeze(-1)  # 1, 256, n
                        cos_eps = 1e-7
                        
                        args.selection_strategy = "s2l_pos_feat_v2"
                        if args.selection_strategy == "s2l_pos_feat_v2":
                            ind_last = valid_indices[-1]
                            q = pro_fg[:, :, :-1].transpose(-2, -1)             # 1, n-1, 256
                            k1 = pro_fg[:, :, -1:]                              # 1, 256, 1, last frame (pos)
                            k2 = t_pos_and_prevs[0][1]["pro_fg"].squeeze(-1)    # 1, 256, 1, first frame (feat)
                            
                            q_norm = torch.norm(q, 2, 2, True)      # 1, n-1, 256
                            k1_norm = torch.norm(k1, 2, 1, True)    # 1, 256, 1
                            k2_norm = torch.norm(k2, 2, 1, True)    # 1, 256, 1
                            
                            cos1 = (q @ k1) / (q_norm @ k1_norm + cos_eps)  # 1, n-1, 1
                            cos2 = (q @ k2) / (q_norm @ k2_norm + cos_eps)  # 1, n-1, 1
                            cos1 = cos1.squeeze(-1)  # 1, n-1
                            cos2 = cos2.squeeze(-1)  # 1, n-1
                            
                            alpha = args.alpha
                            cos = cos1 * alpha + cos2 * (1. - alpha)
                            
                            vals, inds = cos.topk(num_maskmem - 1, dim=1, largest=True, sorted=True)  # 1, num_maskmem - 1
                            inds = inds.squeeze(0)  # num_maskmem - 1
                            inds = inds.sort(descending=False, dim=0)[0]  # sort indices
                            
                            valid_indices = torch.tensor(valid_indices).to(q.device)
                            valid_indices = valid_indices[inds]  # select most similar frames
                            
                            valid_indices = valid_indices.detach().cpu().numpy().tolist()
                            valid_indices.append(ind_last)
                        else:
                            print("The selection strategy should be s2l_pos_feat_v2.")
                            sys.exit(0)
                            
                    self.valid_indices = valid_indices
                    
                args.frame_idx = frame_idx
                args.valid_indices = valid_indices
                
                # ========================================
                # Prepare to process selected candidates
                # ========================================
                start_idx = len(t_pos_and_prevs)
                end_idx = self.num_maskmem
                for t_pos in range(start_idx, end_idx):  # Iterate over the number of mask memories
                    idx = t_pos - self.num_maskmem  # Calculate the index for valid indices
                    if idx < -len(valid_indices):  # Skip if index is out of bounds
                        continue
                    out = output_dict["non_cond_frame_outputs"].get(valid_indices[idx], None)  # Get output for the valid index
                    if out is None:  # If not found, check unselected outputs
                        out = unselected_cond_outputs.get(valid_indices[idx], None)
                    t_pos_and_prevs.append((t_pos, out))  # Append the temporal position and output to the list
            else:
                for t_pos in range(1, self.num_maskmem):
                    t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                    if t_rel == 1:
                        # for t_rel == 1, we take the last frame (regardless of r)
                        if not track_in_reverse:
                            # the frame immediately before this frame (i.e. frame_idx - 1)
                            prev_frame_idx = frame_idx - t_rel
                        else:
                            # the frame immediately after this frame (i.e. frame_idx + 1)
                            prev_frame_idx = frame_idx + t_rel
                    else:
                        # for t_rel >= 2, we take the memory frame from every r-th frames
                        if not track_in_reverse:
                            # first find the nearest frame among every r-th frames before this frame
                            # for r=1, this would be (frame_idx - 2)
                            prev_frame_idx = ((frame_idx - 2) // stride) * stride
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                        else:
                            # first find the nearest frame among every r-th frames after this frame
                            # for r=1, this would be (frame_idx + 2)
                            prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                    out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                    if out is None:
                        # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                        # frames, we still attend to it as if it's a non-conditioning frame.
                        out = unselected_cond_outputs.get(prev_frame_idx, None)
                    t_pos_and_prevs.append((t_pos, out))

            # ========================================
            # Fetch useful information of selected candidates
            # ========================================
            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue
                # memory features
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                
                # spatial positional encoding
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                
                # temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)
                
                # memory mask {0, 1}
                mask = prev["pred_masks"].to(device, non_blocking=True)
                mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)  # (b=1, c=1, h=32, w=32)
                mask = F.sigmoid(mask)
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0
                to_cat_memory_mask.append(mask)
                
                # features
                to_cat_prev_feat.append(prev["image_features"].to(device, non_blocking=True).unsqueeze(1))
                
                # prototypes
                pro_fg = prev["pro_fg"].to(device, non_blocking=True)
                pro_bg = prev["pro_bg"].to(device, non_blocking=True)
                to_cat_pro_fg.append(pro_fg)
                to_cat_pro_bg.append(pro_bg)
            
            # ========================================
            # Object pointers
            # ========================================
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]

                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                        
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]
            to_cat_memory_mask = None
            to_cat_prev_feat = None
            to_cat_pro_fg = None
            to_cat_pro_bg = None
            
        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        memory_mask = torch.cat(to_cat_memory_mask, dim=1) if to_cat_memory_mask is not None else None
        prev_feat = torch.cat(to_cat_prev_feat, dim=1) if to_cat_prev_feat is not None else None  # b, n, c, h, w

        # ========================================
        # Prior-Guided Memory Attention
        # ========================================
        pix_feat_with_mem, prior = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,            # fg memory
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
            memory_mask=memory_mask,  # memory mask with a shape of (b, n, h, w) or None
            pro_fg=to_cat_pro_fg,     # fg prototypes
            pro_bg=to_cat_pro_bg,     # bg prototypes
            args=args,
            return_prior=True
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(-1, C, H, W)
        
        if self.samite_mode and frame_idx > 0:
            if prior is not None:
                if not isinstance(prior, list):
                    prior = F.interpolate(prior, size=(self.image_size, self.image_size), mode='bilinear')
                    prior = prior * 20. - 10.
                else:
                    prior = [F.interpolate(p, size=(self.image_size, self.image_size), mode='bilinear') for p in prior]
                    prior = [p * 20. - 10. for p in prior]
                    prior = [prior[1]]
                    
            return pix_feat_with_mem, prior, prev_feat, memory_mask 
        else:
            return pix_feat_with_mem

    def calculate_IoU(self, output, target, K, ignore_index=255):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.dim() in [1, 2, 3])
        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
        output[target == ignore_index] = ignore_index
        intersection = output[output == target]
        area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
        area_output = torch.histc(output, bins=K, min=0, max=K - 1)
        area_target = torch.histc(target, bins=K, min=0, max=K - 1)
        area_union = area_output + area_target - area_intersection
        return area_intersection, area_union, area_target

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )
        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
        args=None
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                args=args
            )   
            
            if self.samite_mode and frame_idx > 0:
                pix_feat, prior, prev_feat, prev_mask = pix_feat
            else:
                prior = None
                prev_feat = None
                prev_mask = None
            
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                # assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            
            if args is not None and args.use_prior_prompt:
                mask_inputs = prior
            
            # ========================================
            # Batch
            # ========================================
            empty_first = False
            if prior is not None and isinstance(prior, list):
                empty_first = True
                if len(prior) == 1 or len(prior) == 2:
                    # copy pixel feat by 2 times 
                    pix_feat = pix_feat.expand(1 + len(prior), -1, -1, -1).contiguous()  # b=2/3, c, h, w
                    
                    # 3 prior masks
                    prior_0 = [torch.zeros_like(prior[-1])]
                    mask_inputs = torch.cat(prior_0 + prior, dim=0)  # b=2/3, 1, h, w
                
                    # copy each high res feature by 3 times
                    if high_res_features is not None:
                        high_res_features = [feat.expand(1 + len(prior), -1, -1, -1) for feat in high_res_features]
                else:
                    print("Currently only support bs 2.")
                    sys.exit(0)
            elif prior is not None and not isinstance(prior, list):  # post calibration
                print("Current only support prior list.")
                sys.exit(0)
                            
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            
            with torch.no_grad():
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    empty_first=empty_first,
                    prev_feat=prev_feat,
                    prev_mask=prev_mask,
                    args=args,
                    prior=prior,
                    update_kalman=True
                )
                    
        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            # fg memory
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            # fg memory
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None
            
    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
        args=None
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
            args=args
        )

        if len(sam_outputs) == 10:
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                best_iou_score,
                kf_ious,
                _,
            ) = sam_outputs
        else:
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = sam_outputs
            best_iou_score = 1.0
            kf_ious = None

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["best_iou_score"] = best_iou_score
        current_out["kf_ious"] = kf_ious
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out
        )
            
        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
