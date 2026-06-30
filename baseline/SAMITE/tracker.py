import numpy as np
import yaml
import torch
import torchvision.transforms.functional as F
from sam2.build_sam import build_sam2_video_predictor
from collections import OrderedDict
import random
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.utils import keep_largest_component, determine_tracker

from pathlib import Path
config_path = Path(
    os.environ.get("SAM2_CONFIG", str(REPO_ROOT / "config.yaml"))
)
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


class Args:
    def __init__(self):
        self.max_num_indices = 30 # Number of candidate previous frames
        self.selection_strategy = "s2l_pos_feat_v2", # choices=["none", "s2l_pos_feat_v2"], help="Frame selection strategy")
        self.bias_mode = "kalman_pos" # choices=["none", "kalman_pos",], help="Mode of Cross Attention Bias")
        self.bias_type = "v3" # choices=["v3", "none"], help="Type of Cross Attention Bias")
        self.use_prior_prompt =False
        self.alpha = 0.3 # help="Fusion weight in frame selection."
    

class SAMTracker():
    def __init__(self, tracker_name="sam21-L"):
        """
        Constructor for the SAM (2 or 2.1) tracking wrapper.

        Args:
        - tracker_name (str): Name of the tracker to use. Options are:
            - "sam21-L": SAM2.1 Hiera Large
            - "sam21-B": SAM2.1 Hiera Base+
            - "sam21-S": SAM2.1 Hiera Small
            - "sam21-T": SAM2.1 Hiera Tiny
            - "sam2-L": SAM2 Hiera Large
            - "sam2-B": SAM2 Hiera Base+
            - "sam2-S": SAM2 Hiera Small
            - "sam2-T": SAM2 Hiera Tiny
        """
        self.checkpoint, self.model_cfg = determine_tracker(tracker_name)
        self.device = config.get("device", "cuda:0")

        # Image preprocessing parameters
        self.input_image_size = 1024       
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=self.device)
        self.tracking_times = []

        self.args = Args()

    def _prepare_image(self, img_pil):
        # _load_img_as_tensor from SAM2
        img = torch.from_numpy(np.array(img_pil)).to(self.inference_state["device"])
        img = img.permute(2, 0, 1).float() / 255.0
        img = F.resize(img, (self.input_image_size, self.input_image_size))
        img = (img - self.img_mean) / self.img_std        
        return img

    @torch.inference_mode()
    def init_state_tw(
        self,
    ):
        """Initialize an inference state."""
        compute_device = torch.device(self.device)
        inference_state = {}
        inference_state["images"] = None # later add, step by step
        inference_state["num_frames"] = 0 # later add, step by step
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = False
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = False
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = None # later add, step by step
        inference_state["video_width"] =  None # later add, step by step
        inference_state["device"] = compute_device
        inference_state["storage_device"] = compute_device #torch.device("cpu")
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        inference_state["frames_tracked_per_obj"] = {}
        
        self.img_mean = self.img_mean.to(compute_device)
        self.img_std = self.img_std.to(compute_device)

        return inference_state
    
    @torch.inference_mode()
    def initialize(self, image, init_mask, bbox=None):
        """
        Initialize the tracker with the first frame and mask.
        Function builds the SAM (2 or 2.1) tracker and initializes it with the first frame and mask.

        Args:
        - image (PIL Image): First frame of the video.
        - init_mask (numpy array): Binary mask for the initialization
        
        Returns:
        - out_dict (dict): Dictionary containing the mask for the initialization frame.
        """
        if type(init_mask) is list:
            init_mask = init_mask[0]
        self.frame_index = 0 # Current frame index, updated frame-by-frame
        self.object_sizes = [] # List to store object sizes (number of pixels) 
        self.last_added = -1 # Frame index of the last added frame into DRM memory
        
        self.img_width = image.width # Original image width
        self.img_height = image.height # Original image height
        self.inference_state = self.init_state_tw()
        self.inference_state["images"] = image
        video_width, video_height = image.size 
        self.inference_state["video_height"] = video_height
        self.inference_state["video_width"] =  video_width
        prepared_img = self._prepare_image(image)
        self.inference_state["images"] = {0 : prepared_img}
        self.inference_state["num_frames"] = 1
        self.predictor.reset_state(self.inference_state)

        # warm up the model
        self.predictor._get_image_feature(self.inference_state, frame_idx=0, batch_size=1)

        if init_mask is None:
            if bbox is None:
                print('Error: initialization state (bbox or mask) is not given.')
                exit(-1)
            
            # consider bbox initialization - estimate init mask from bbox first
            init_mask = self.estimate_mask_from_box(bbox)


        # _, _, out_mask_logits = self.predictor.add_new_mask(
            # inference_state=self.inference_state,
            # frame_idx=0,
            # obj_id=0,
            # mask=init_mask,
        # )

        bbox_xyxy = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])[None, :]
        _, _, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            box=bbox_xyxy, 
            frame_idx=0, 
            obj_id=0
            )

        m = (out_mask_logits[0, 0] > 0).float().cpu().numpy().astype(np.uint8)
        self.inference_state["images"].pop(self.frame_index)

        out_dict = {'pred_mask': m}
        return out_dict

    @torch.inference_mode()
    def track(self, image, init=False):
        """
        Function to track the object in the next frame.

        Args:
        - image (PIL Image): Next frame of the video.
        - init (bool): Whether the current frame is the initialization frame.

        Returns:
        - out_dict (dict): Dictionary containing the predicted mask for the current frame.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Prepare the image for input to the model
        prepared_img = self._prepare_image(image)
        if not init:
            self.frame_index += 1
            self.inference_state["num_frames"] += 1
        self.inference_state["images"][self.frame_index] = prepared_img

        # Propagate the tracking to the next frame
        # return_all_masks=True returns all predicted (chosen and alternative) masks and corresponding IoUs
        for out in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=self.frame_index, max_frame_num_to_track=0, args=self.args):
            # There are 3 outputs when the tracking is done on the frame
            out_frame_idx, _, out_mask_logits = out
            m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)

            # Return the predicted mask for the current frame
            out_dict = {'pred_mask': m}

            self.inference_state["images"].pop(self.frame_index)
            return out_dict

    def estimate_mask_from_box(self, bbox):
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self.predictor._get_image_feature(self.inference_state, 0, 1)

        box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])[None, :]
        box = torch.as_tensor(box, dtype=torch.float, device=current_vision_feats[0].device)

        from sam2.utils.transforms import SAM2Transforms
        _transforms = SAM2Transforms(
            resolution=self.predictor.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        unnorm_box = _transforms.transform_boxes(
            box, normalize=True, orig_hw=(self.img_height, self.img_width)
        )  # Bx2x2
        
        box_coords = unnorm_box.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=unnorm_box.device)
        box_labels = box_labels.repeat(unnorm_box.size(0), 1)
        concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=None
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = []
        for i in range(2):
            _, b_, c_ = current_vision_feats[i].shape
            high_res_features.append(current_vision_feats[i].permute(1, 2, 0).view(b_, c_, feat_sizes[i][0], feat_sizes[i][1]))
        if self.predictor.directly_add_no_mem_embed:
            img_embed = current_vision_feats[2] + self.predictor.no_mem_embed
        else:
            img_embed = current_vision_feats[2]
        _, b_, c_ = current_vision_feats[2].shape
        img_embed = img_embed.permute(1, 2, 0).view(b_, c_, feat_sizes[2][0], feat_sizes[2][1])
        low_res_masks, iou_predictions, _, _ = self.predictor.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = _transforms.postprocess_masks(
            low_res_masks, (self.img_height, self.img_width)
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        masks = masks > 0

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()

        init_mask = masks_np[0]
        return init_mask
