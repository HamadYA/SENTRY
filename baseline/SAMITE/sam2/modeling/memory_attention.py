# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import os
import sys
import cv2
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones


class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {
                "num_k_exclude_rope": num_k_exclude_rope
            }

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,  # fg memory
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
        
        # ========================================
        # Positional Encoding
        # ========================================
        self.pe = PositionalEncodingPermute2D(channels=256)

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
        memory_mask: Optional[Tensor] = None,  # to support memory mask
        pro_fg=None,  # fg prototypes
        pro_bg=None,  # bg prototypes
        args=None,  # additional parameters
        return_prior=False
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        q = output.clone()
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            q = q.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        # ========================================
        # Cross Attention Bias
        # - if bias_type is not v1 or v2
        # ========================================
        priors_ = None
        h = w = output.size(1) ** 0.5
        fts_size = (int(h), int(w))
        if args.bias_type in ["v3"]:
            assert pro_fg is not None and pro_bg is not None, "When bias_type is not v3, please extract fg and bg prototypes from original features."
            assert memory_mask is not None, "Please ensure memory_mask is available."
            
            
            q = rearrange(output, 'b (h w) c -> b c h w', h=int(h), w=int(w))  # b, c, h, w
            enc = self.pe(q)
            
            if args.bias_mode == "kalman_pos":
                enc_priors = self.generate_prior_enc(enc, enc, [memory_mask[:, -1:, ...]], fts_size=fts_size)[0]
                _, _, priors_1, priors_2 = self.generate_prior_feat_enc_batch(
                    q, pro_fg, pro_bg, enc_priors, fts_size=fts_size
                )  # b, n, h, w
                
                priors_ = [priors_1.mean(1, True), priors_2.mean(1, True)]  # check_results()
            else:
                print("Bias mode should be kalman_pos")
                sys.exit(0)
        else:
            print("Bias type should be v3")
            sys.exit(0)

        for idx, layer in enumerate(self.layers):
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                args.layer_idx = idx
                kwds = {
                    "num_k_exclude_rope": num_obj_ptr_tokens,
                }

            output = layer(
                tgt=output,
                memory=memory,  # fg memory
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        if not return_prior:
            return normed_output
        else:
            return normed_output, priors_
   
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
 
    def generate_prior_enc(self, qry_feat, sup_feat, sup_masks, fts_size):
        bsize, ch_sz, sp_sz, _ = qry_feat.size()[:]
        cos_eps = 1e-7
        
        sim_fgs = []
        for sup_mask in sup_masks:
            # resize mask
            resize_size = sup_feat.size()[-2:]
            sup_mask = F.interpolate(sup_mask, size=resize_size, mode='bilinear', align_corners=False)  # b, 1, h, w

            # fg prototype
            sup_fg = self.Weighted_GAP(sup_feat, sup_mask)
            
            # cosine similarities
            sim_fg = self.cos_sim(qry_feat, sup_fg, cos_eps)
            
            # fg and bg priors, normalize to [0, 1]
            sim_fg = sim_fg.max(1)[0].view(bsize, sp_sz * sp_sz)
            sim_fg = (sim_fg - sim_fg.min(1)[0].unsqueeze(1)) / (
                        sim_fg.max(1)[0].unsqueeze(1) - sim_fg.min(1)[0].unsqueeze(1) + cos_eps)
            sim_fg = sim_fg.view(bsize, 1, sp_sz, sp_sz)

            # interpolate
            sim_fg = F.interpolate(sim_fg, size=fts_size, mode='bilinear', align_corners=False)
            
            sim_fgs.append(sim_fg)
        
        return sim_fgs
 
    def generate_prior_feat_enc_batch(self, qry_feat, sup_fgs, sup_bgs, enc_priors, fts_size, cycle=False):
        '''
        Warning!!!
        Note that this function has already been modified, please refer to that in memory_attention_backup for the original one.
        '''
        bsize, ch_sz, sp_sz, _ = qry_feat.size()[:]
        cos_eps = 1e-7
        
        sup_fg = torch.cat(sup_fgs, dim=2) if isinstance(sup_fgs, list) else sup_fgs  # b, c, n, 1
        sup_bg = torch.cat(sup_bgs, dim=2) if isinstance(sup_bgs, list) else sup_bgs
        enc_prior = torch.cat(enc_priors, dim=1) if isinstance(enc_priors, list) else enc_priors
        num = sup_fg.size(-2)
        
        # ========================================
        # first prior (without scond norm)
        # ========================================
        # cos sim
        sim_fg = self.cos_sim(qry_feat, sup_fg, cos_eps)  # b, n-1, hw
        sim_bg = self.cos_sim(qry_feat, sup_bg, cos_eps)
        
        # # local norm (within each previous frame)
        # sim_fg = (sim_fg - sim_fg.min(-1)[0].unsqueeze(-1)) / (sim_fg.max(-1)[0].unsqueeze(-1) - sim_fg.min(-1)[0].unsqueeze(-1) + cos_eps)
        # sim_bg = (sim_bg - sim_bg.min(-1)[0].unsqueeze(-1)) / (sim_bg.max(-1)[0].unsqueeze(-1) - sim_bg.min(-1)[0].unsqueeze(-1) + cos_eps)
        # # sim_fg = (sim_fg + 1.) / 2.
        # # sim_bg = (sim_bg + 1.) / 2.
        # sim_fg = rearrange(sim_fg, 'b n (h w) -> b n h w', h=sp_sz)
        # sim_bg = rearrange(sim_bg, 'b n (h w) -> b n h w', h=sp_sz)

        # # disc prior - more similar to fg than bg
        # sim_disc = (sim_fg - sim_bg)  # b, n, h, w
 
        # # the only difference lies in the position of using enc_prior (kalman info)
        # if not cycle:
        #     # output 1 - prior without second norm
        #     sim_disc[sim_disc < 0] = 0
        #     sim_disc_1 = sim_disc.clone()
        #     sim_disc_1 = rearrange(sim_disc_1, 'b n h w -> b n (h w)')
        #     sim_disc_1 = (sim_disc_1 - 0) / (sim_disc_1.max(-1)[0].unsqueeze(-1) - 0 + cos_eps)
        #     sim_disc_1 = rearrange(sim_disc_1, 'b n (h w) -> b n h w', h=sp_sz)
        #     sim_disc_1 = F.interpolate(sim_disc_1, size=fts_size, mode='bilinear', align_corners=False)
    
        #     # ========================================
        #     # second prior (with second norm)
        #     # ========================================
        #     # output 2 - prior with second local norm (within each previous frame)
        #     sim_disc = sim_disc * enc_prior
        #     sim_disc = rearrange(sim_disc, 'b n h w -> b n (h w)')
        #     sim_disc = (sim_disc - 0) / (sim_disc.max(-1)[0].unsqueeze(-1) - 0 + cos_eps)
        #     sim_disc = rearrange(sim_disc, 'b n (h w) -> b n h w', h=sp_sz)
        #     sim_disc_2 = F.interpolate(sim_disc, size=fts_size, mode='bilinear', align_corners=False)
        # else:
        #     # output 1 - prior without second norm
        #     sim_disc[sim_disc < 0] = 0  # do not use enc_prior at this moment
        #     sim_disc_1 = F.interpolate(sim_disc, size=fts_size, mode='bilinear', align_corners=False)
            
        #     # output 2 - prior with second local norm (within each previous frame)
        #     sim_disc = rearrange(sim_disc, 'b n h w -> b n (h w)')
        #     sim_disc = (sim_disc - 0) / (sim_disc.max(-1)[0].unsqueeze(-1) - 0 + cos_eps)
        #     sim_disc = rearrange(sim_disc, 'b n (h w) -> b n h w', h=sp_sz)
        #     sim_disc = sim_disc * enc_prior
        #     sim_disc_2 = F.interpolate(sim_disc, size=fts_size, mode='bilinear', align_corners=False)
        
        # local norm (within each previous frame)
        sim_fg_1 = (sim_fg - sim_fg.min(-1)[0].unsqueeze(-1)) / (sim_fg.max(-1)[0].unsqueeze(-1) - sim_fg.min(-1)[0].unsqueeze(-1) + cos_eps)
        sim_bg_1 = (sim_bg - sim_bg.min(-1)[0].unsqueeze(-1)) / (sim_bg.max(-1)[0].unsqueeze(-1) - sim_bg.min(-1)[0].unsqueeze(-1) + cos_eps)
        sim_fg_2 = (sim_fg + 1.) / 2.
        sim_bg_2 = (sim_bg + 1.) / 2.
        
        sim_fg_1 = rearrange(sim_fg_1, 'b n (h w) -> b n h w', h=sp_sz)
        sim_bg_1 = rearrange(sim_bg_1, 'b n (h w) -> b n h w', h=sp_sz)
        sim_fg_2 = rearrange(sim_fg_2, 'b n (h w) -> b n h w', h=sp_sz)
        sim_bg_2 = rearrange(sim_bg_2, 'b n (h w) -> b n h w', h=sp_sz)

        # disc prior - more similar to fg than bg
        sim_disc_1 = (sim_fg_1 - sim_bg_1)  # b, n, h, w
        sim_disc_2 = (sim_fg_2 - sim_bg_2)  # b, n, h, w
 
        # # output 1 - minmax norm
        # sim_disc_1[sim_disc_1 < 0] = 0
        # sim_disc_1 = sim_disc_1 * enc_prior
        # sim_disc_1 = rearrange(sim_disc_1, 'b n h w -> b n (h w)')
        # sim_disc_1 = (sim_disc_1 - 0) / (sim_disc_1.max(-1)[0].unsqueeze(-1) - 0 + cos_eps)
        # sim_disc_1 = rearrange(sim_disc_1, 'b n (h w) -> b n h w', h=sp_sz)
        # sim_disc_1 = F.interpolate(sim_disc_1, size=fts_size, mode='bilinear', align_corners=False)

        # output 2 - shift norm
        sim_disc_2[sim_disc_2 < 0] = 0
        
        sim_disc_1 = sim_disc_2.clone()
        sim_disc_1 = rearrange(sim_disc_1, 'b n h w -> b n (h w)')
        sim_disc_1 = (sim_disc_1 - 0) / (sim_disc_1.max(-1)[0].unsqueeze(-1) - 0 + cos_eps)
        sim_disc_1 = rearrange(sim_disc_1, 'b n (h w) -> b n h w', h=sp_sz)
        sim_disc_1 = F.interpolate(sim_disc_1, size=fts_size, mode='bilinear', align_corners=False)
        
        sim_disc_2 = sim_disc_2 * enc_prior
        sim_disc_2 = rearrange(sim_disc_2, 'b n h w -> b n (h w)')
        sim_disc_2 = (sim_disc_2 - 0) / (sim_disc_2.max(-1)[0].unsqueeze(-1) - 0 + cos_eps)
        sim_disc_2 = rearrange(sim_disc_2, 'b n (h w) -> b n h w', h=sp_sz)
        sim_disc_2 = F.interpolate(sim_disc_2, size=fts_size, mode='bilinear', align_corners=False)
            
        # return sim_fg, sim_bg, sim_disc_1, sim_disc_2
        return sim_fg_1, sim_bg_1, sim_disc_1, sim_disc_2

    def Weighted_GAP(self, supp_feat, mask):
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        return supp_feat

    def cos_sim(self, qry_feat, sup_feat, cos_eps=1e-7):
        q = qry_feat.flatten(2).transpose(-2, -1)
        s = sup_feat.flatten(2).transpose(-2, -1)

        qry = q
        qry = qry.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
        qry_norm = torch.norm(qry, 2, 1, True)

        sup = s
        sup = sup.contiguous()
        if sup.size(1) == 1:
            sup_norm = torch.norm(sup, 2, 2, True)
        else:
            sup_norms = []
            for i in range(sup.size(1)):
                sup_norm = torch.norm(sup[:, i:i+1, :], 2, 2, True)
                sup_norms.append(sup_norm)
            sup_norm = torch.cat(sup_norms, dim=1)  # b, n, c

        cos = torch.bmm(sup, qry) / (torch.bmm(sup_norm, qry_norm) + cos_eps)
        return cos

    
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
        tensor = tensor.permute(0, 2, 3, 1)  # b, h, w, c
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)  # b, c, h, w

    @property
    def org_channels(self):
        return self.penc.org_channels
    