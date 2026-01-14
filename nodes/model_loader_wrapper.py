"""
DreamID-V Model Loader Wrapper
Direct path loading without temporary directories
Updated to support 'Faster' backend with Monkey Patched Generate
Fixed: 
1. Strict separate handling of ID Image (img_ref) and Video/Mask (y).
2. Smart VAE Channel validation (Warns if 4-channel SD VAE is detected).
3. CRITICAL FIX: Force Reference Image to match Video Latent dimensions (Fit & Pad) to prevent RoPE shape mismatch.
"""

import torch
import torch.nn.functional as F
import os
import folder_paths
import logging
import math
import sys
import random
from easydict import EasyDict
from torchvision.transforms import Normalize, Compose
from contextlib import contextmanager
from tqdm import tqdm
import torch.cuda.amp as amp

# Import original components
from ..dreamidv_wan.modules.model import WanModel
from ..dreamidv_wan.modules.vae import WanVAE
from ..dreamidv_wan.modules.t5 import T5EncoderModel
from ..dreamidv_wan.configs import WAN_CONFIGS

# Import Faster components
FASTER_AVAILABLE = False
try:
    from ..dreamidv_wan_faster.configs import WAN_CONFIGS as WAN_CONFIGS_FASTER
    from ..dreamidv_wan_faster.wan_swapface import DreamIDV as DreamIDV_Faster
    from ..dreamidv_wan_faster.utils.na_resize import NaResize, DivisibleCrop, Rearrange
    from ..dreamidv_wan_faster.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    # Use Faster version of WanModel (No Pose Projector)
    from ..dreamidv_wan_faster.modules.model import WanModel as WanModel_Faster
    FASTER_AVAILABLE = True
except ImportError:
    logging.warning("[DreamID-V Wrapper] 'dreamidv_wan_faster' folder incomplete or missing. Faster backend will be unavailable.")
    from ..dreamidv_wan.utils.na_resize import NaResize, DivisibleCrop, Rearrange

# --- Helper: Try to find VAE Name ---
def get_vae_name(vae_instance):
    try:
        if hasattr(vae_instance, "patcher") and hasattr(vae_instance.patcher, "load_name"):
            return vae_instance.patcher.load_name
    except:
        pass
    return "Unknown"

# --- Patch 1: Faster Data Loader (Smart Shape & Channel Handling + Padding) ---
def faster_load_data_tensor(self, paths, size, device, frame_num):
    """
    Adapter to load Tensors directly for Faster model
    Ensures Image Latents match Video Latent dimensions exactly via Fit & Pad.
    """
    patch_size = self.patch_size
    vae_stride = self.vae_stride

    ref_video = paths[0]
    mask_video = paths[1]
    ref_image = paths[2]

    ref_vae_latents = {}
    resolution = math.sqrt(size[0] * size[1])
    
    # Standard Transform for Video
    resize_crop = Compose([
        NaResize(resolution=resolution, downsample_only=True),
        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
    ])
    rearrange = Rearrange("t c h w -> c t h w")
    norm = Normalize(0.5, 0.5)

    # --- DEBUG VAE ---
    if not hasattr(self, "_vae_debug_printed"):
        vae_name = get_vae_name(self.vae.comfy_vae) if hasattr(self.vae, "comfy_vae") else "Unknown"
        logging.info("="*50)
        logging.info(f"[DreamID-V VAE INSPECTOR]")
        logging.info(f"  > Loaded VAE Name: {vae_name}")
        self._vae_debug_printed = True

    # 1. Process Video First (Master Dimensions)
    # ------------------------------------------------
    video_frames = ref_video.permute(0, 3, 1, 2) # [T, C, H, W]
    
    # Frame slicing
    video_frames = video_frames[:frame_num]
    valid_frames = (len(video_frames) - 1) // 4 * 4 + 1
    video_frames = video_frames[:valid_frames]
    
    # Apply Transform to Video
    video_processed = resize_crop(video_frames) # [T, C, H, W]
    
    # Capture Target Spatial Dimensions (H, W) for Image Padding
    target_h, target_w = video_processed.shape[2], video_processed.shape[3]
    
    if hasattr(self, "_vae_debug_printed"):
        logging.info(f"  > Video Target Shape: {target_h}x{target_w}")

    # Norm & Rearrange Video
    video_processed = norm(video_processed)
    video_processed = rearrange(video_processed) # -> [C, T, H, W]
    
    # Encode Video
    encoded_video = self.vae.encode([video_processed], device)[0]
    
    # 2. Process Mask (Force Match Video Size)
    # ------------------------------------------------
    mask_frames = mask_video.permute(0, 3, 1, 2)
    mask_frames = mask_frames[:valid_frames] # Sync length
    
    # Resize Mask using Interpolation to strictly match Video shape
    # (Avoids rounding diffs from NaResize)
    mask_frames = F.interpolate(mask_frames, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    # Mask usually doesn't strictly need Norm -1..1 but let's follow standard
    # mask_frames = norm(mask_frames) # Optional based on original logic
    mask_processed = rearrange(mask_frames)
    encoded_mask = self.vae.encode([mask_processed], device)[0]

    # 3. Process Reference Image (Fit & Pad Logic)
    # ------------------------------------------------
    # JR Logic: Resize to fit inside target box, then pad to fill.
    img_tensor = ref_image.permute(0, 3, 1, 2) # [1, C, H, W]
    
    curr_h, curr_w = img_tensor.shape[2], img_tensor.shape[3]
    img_ratio = curr_w / curr_h
    target_ratio = target_w / target_h
    
    if img_ratio > target_ratio:
        # Fit Width
        new_w = target_w
        new_h = int(new_w / img_ratio)
    else:
        # Fit Height
        new_h = target_h
        new_w = int(new_h * img_ratio)
        
    # 1. Resize (Fit)
    img_resized = F.interpolate(img_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
    
    # 2. Pad (Fill)
    # Calculate padding [left, right, top, bottom]
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    
    # Apply padding (Color 0 = gray in -1..1 space? Or 255 white?)
    # JR uses fill=(255,255,255). In Tensor 0..1, that is 1.0.
    # We pad with constant 1.0 (White background like JR)
    img_padded = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=1.0)
    
    # Norm & Rearrange
    img_processed = norm(img_padded)
    
    # Permute for VAE: [1, C, H, W] -> [C, 1, H, W] (Rearrange usually expects T)
    img_processed = img_processed.permute(1, 0, 2, 3) 
    
    encoded_image = self.vae.encode([img_processed], device)[0]

    # --- HELPER: Shape Validation ---
    def validate_and_fix(tensor, name, is_single_frame=False):
        # Expected: [16, T, H, W]
        # Check Channels
        if tensor.dim() == 3: tensor = tensor.unsqueeze(1)
        
        c_dim = -1
        if tensor.shape[0] == 16: c_dim = 0
        elif tensor.shape[1] == 16: c_dim = 1
        
        if c_dim == -1:
            logging.error(f"[DreamID-V ERROR] Input '{name}' has wrong channels (Expected 16). Shape: {tensor.shape}")
            # Try 4->16 hack if needed, or fail.
            if tensor.shape[1] == 4: tensor = tensor.permute(1, 0, 2, 3)
        elif c_dim == 1:
            tensor = tensor.permute(1, 0, 2, 3)
            
        logging.info(f"  > {name} Final Latent: {tensor.shape} (✅)")
        return tensor

    ref_vae_latents["video"] = validate_and_fix(encoded_video, "ref_video")
    ref_vae_latents["mask"] = validate_and_fix(encoded_mask, "mask_video")
    ref_vae_latents["image"] = validate_and_fix(encoded_image, "ref_image")
    
    if hasattr(self, "_vae_debug_printed"):
        logging.info("="*50)

    return ref_vae_latents

# --- Patch 2: Faster Generate (Strict Separation) ---
def faster_generate_tensor(self,
                           input_prompt,
                           paths,
                           size=(1280, 720),
                           frame_num=81,
                           shift=5.0,
                           sample_solver='unipc',
                           sampling_steps=12,
                           guide_scale_img=5.0,
                           seed=-1,
                           offload_video_model=True,
                           offload_t5=True,
                           return_latent=True,
                           **kwargs):
    
    device = self.device
    dtype = self.param_dtype
    
    latents_ref = self.load_image_latent_ref_ip_video(paths, size, device, frame_num)
    
    latents_ref_video = latents_ref["video"].to(device, dtype)
    latents_ref_image = latents_ref["image"].to(device, dtype)
    msk = latents_ref["mask"].to(device, dtype)

    target_shape = (16, 
                    latents_ref_video.shape[1],
                    latents_ref_video.shape[2],
                    latents_ref_video.shape[3])

    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    plugin_dir = os.path.dirname(os.path.dirname(__file__)) 
    context_path = os.path.join(plugin_dir, "dreamidv_wan_faster", "context.pth")
    if not os.path.exists(context_path):
        raise FileNotFoundError(f"Faster context missing: {context_path}")

    context = torch.load(context_path, map_location="cpu")
    context = [t.to(self.device) for t in context]

    img_ref = latents_ref_image
    
    # Concatenate Video(16) + Mask(16) -> 32
    y_concat = torch.cat([latents_ref_video, msk], dim=0)
    
    arg_tiv = {
        'context': context, 
        'seq_len': seq_len,
        'y': [y_concat],
        'img_ref': [img_ref]
    }
    
    arg_tv = {
        'context': context,
        'seq_len': seq_len, 
        'y': [y_concat],
        'img_ref': [torch.zeros_like(img_ref)]
    }

    noise = [
        torch.randn(
            target_shape[0], target_shape[1], target_shape[2], target_shape[3],
            dtype=torch.float32, device=self.device, generator=seed_g)
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1, use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
        timesteps = sample_scheduler.timesteps

        latents = noise

        for i, t in enumerate(tqdm(timesteps)):
            timestep = [t]
            timestep = torch.stack(timestep)
            self.model.to(self.device)
            
            pos_tiv = self.model(latents, t=timestep, **arg_tiv)[0]
            pos_tv = self.model(latents, t=timestep, **arg_tv)[0]
            
            noise_pred = pos_tiv + guide_scale_img * (pos_tiv - pos_tv) 
            
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]

        if offload_video_model:
            self.model.cpu()
            torch.cuda.empty_cache()

        if return_latent:
            return latents[0] 
        
        videos = self.vae.decode(latents)
        return videos[0]


class DreamIDV_ModelLoader_Wrapper_TTP:
    _cached_wrapper = None
    _cached_paths = None
    
    @classmethod
    def INPUT_TYPES(cls):
        dreamidv_models = folder_paths.get_filename_list("diffusion_models")
        t5_files = folder_paths.get_filename_list("text_encoders")
        
        return {
            "required": {
                "dreamidv_model": (dreamidv_models, {"default": "dreamidv.pth"}),
                "backend": (["original", "faster"], {"default": "original"}),
                "t5_checkpoint": (t5_files, {"default": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "quantization": (["disabled", "fp8_e4m3fn", "fp8_e5m2"], {"default": "disabled"}),
            },
            "optional": {
                "attention_mode": (["auto", "flash_attn", "sageattn", "sdpa"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("DREAMIDV_WRAPPER",)
    RETURN_NAMES = ("dreamidv_wrapper",)
    FUNCTION = "load_models"
    CATEGORY = "DreamID-V/Wrapper"
    
    def load_models(self, dreamidv_model, backend, t5_checkpoint, precision="bf16", quantization="disabled", attention_mode="sdpa"):
        if backend == "faster" and not FASTER_AVAILABLE:
            raise RuntimeError("Faster backend missing.")

        dreamidv_path = folder_paths.get_full_path("diffusion_models", dreamidv_model)
        t5_path = folder_paths.get_full_path("text_encoders", t5_checkpoint)
        current_paths = (dreamidv_path, t5_path, precision, quantization, attention_mode, backend)
        
        if (DreamIDV_ModelLoader_Wrapper_TTP._cached_wrapper is not None and
            DreamIDV_ModelLoader_Wrapper_TTP._cached_paths == current_paths):
            return (DreamIDV_ModelLoader_Wrapper_TTP._cached_wrapper,)
        
        if backend == "faster":
            cfg = WAN_CONFIGS_FASTER['swapface']
            ModelClass = WanModel_Faster
            logging.info(f"[DreamID-V Wrapper] Backend: FASTER")
        else:
            cfg = WAN_CONFIGS['swapface']
            ModelClass = WanModel
            logging.info(f"[DreamID-V Wrapper] Backend: ORIGINAL")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(precision, torch.bfloat16)

        class DreamIDVWrapper: pass
        wrapper = DreamIDVWrapper()
        wrapper.device = device
        wrapper.config = cfg
        wrapper.rank = 0
        wrapper.t5_cpu = False
        wrapper.num_train_timesteps = cfg.num_train_timesteps
        wrapper.param_dtype = base_dtype
        wrapper.vae_stride = cfg.vae_stride
        wrapper.patch_size = cfg.patch_size
        wrapper.sp_size = 1
        
        # T5 Loading
        logging.info(f"[DreamID-V Wrapper] Loading T5...")
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        tokenizer_dir = os.path.join(plugin_dir, 'tokenizer_configs')

        if t5_checkpoint.endswith('.safetensors'):
            from safetensors.torch import load_file
            t5_state_dict = load_file(t5_path)
        else:
            t5_state_dict = torch.load(t5_path, map_location='cpu')
        
        is_hf_format = any('encoder.block' in k for k in t5_state_dict.keys())
        
        if is_hf_format:
            is_scaled_fp8 = "scaled_fp8" in t5_state_dict
            scale_weights = {}
            if is_scaled_fp8:
                 for k, v in t5_state_dict.items():
                    if k.endswith(".scale_weight"): scale_weights[k] = v.to('cpu', cfg.t5_dtype)
            converted_sd = {}
            for key, value in t5_state_dict.items():
                if key.endswith(".scale_weight"): continue
                new_key = key
                if key.startswith('encoder.block.'):
                    parts = key.split('.')
                    block_num = parts[2]
                    if 'layer.0.SelfAttention' in key:
                        if key.endswith('.k.weight'): new_key = f"blocks.{block_num}.attn.k.weight"
                        elif key.endswith('.o.weight'): new_key = f"blocks.{block_num}.attn.o.weight"
                        elif key.endswith('.q.weight'): new_key = f"blocks.{block_num}.attn.q.weight"
                        elif key.endswith('.v.weight'): new_key = f"blocks.{block_num}.attn.v.weight"
                        elif 'relative_attention_bias' in key: new_key = f"blocks.{block_num}.pos_embedding.embedding.weight"
                    elif 'layer.0.layer_norm' in key: new_key = f"blocks.{block_num}.norm1.weight"
                    elif 'layer.1.layer_norm' in key: new_key = f"blocks.{block_num}.norm2.weight"
                    elif 'layer.1.DenseReluDense' in key:
                        if 'wi_0' in key: new_key = f"blocks.{block_num}.ffn.gate.0.weight"
                        elif 'wi_1' in key: new_key = f"blocks.{block_num}.ffn.fc1.weight"
                        elif 'wo' in key: new_key = f"blocks.{block_num}.ffn.fc2.weight"
                elif key == "shared.weight": new_key = "token_embedding.weight"
                elif key == "encoder.final_layer_norm.weight": new_key = "norm.weight"
                converted_sd[new_key] = value
                if f"{key}.scale_weight" in scale_weights:
                    scale_weights[f"{new_key}.scale_weight"] = scale_weights.pop(f"{key}.scale_weight")

            from ..dreamidv_wan.modules.t5 import umt5_xxl
            t5_model = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=cfg.t5_dtype, device=torch.device('cpu')).eval().requires_grad_(False)
            t5_model.load_state_dict(converted_sd, strict=False)

            if is_scaled_fp8 and len(scale_weights) > 0:
                logging.info(f"[DreamID-V Wrapper] Applying FP8 linear optimization...")
                def fp8_linear_forward(linear_module, base_dtype, input_tensor):
                    if len(input_tensor.shape) == 3:
                        scale_weight = getattr(linear_module, 'scale_weight', torch.ones((), device=input_tensor.device))
                        input_tensor = torch.clamp(input_tensor, min=-448, max=448, out=input_tensor)
                        inn = input_tensor.reshape(-1, input_tensor.shape[2]).to(torch.float8_e4m3fn).contiguous()
                        bias = linear_module.bias.to(base_dtype) if linear_module.bias is not None else None
                        o = torch._scaled_mm(inn, linear_module.weight.t(), out_dtype=base_dtype, bias=bias, scale_a=torch.ones((), device=input_tensor.device), scale_b=scale_weight)
                        return o.reshape((-1, input_tensor.shape[1], linear_module.weight.shape[0]))
                    return linear_module.original_forward(input_tensor)
                import torch.nn as nn
                for name, module in t5_model.named_modules():
                    if isinstance(module, nn.Linear) and "embedding" not in name:
                        scale_key = f"{name}.scale_weight"
                        if scale_key in scale_weights:
                            setattr(module, "scale_weight", scale_weights[scale_key].float())
                            setattr(module, "original_forward", module.forward)
                            setattr(module, "forward", lambda inp, m=module: fp8_linear_forward(m, cfg.t5_dtype, inp))
        else:
            from ..dreamidv_wan.modules.t5 import umt5_xxl
            t5_model = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=cfg.t5_dtype, device=torch.device('cpu')).eval().requires_grad_(False)
            t5_model.load_state_dict(t5_state_dict)

        import sentencepiece as spm
        spiece_model_path = os.path.join(tokenizer_dir, 'spiece.model')
        if not os.path.exists(spiece_model_path):
            try:
                import urllib.request
                urllib.request.urlretrieve("https://huggingface.co/google/umt5-xxl/resolve/main/spiece.model", spiece_model_path)
            except: pass
        sp = spm.SentencePieceProcessor()
        sp.load(spiece_model_path)

        class T5EncoderWrapper:
            def __init__(self, model, sp_processor):
                self.model = model; self.sp = sp_processor; self.text_len = cfg.text_len
            def __call__(self, texts, device):
                if isinstance(texts, str): texts = [texts]
                ids_list = []
                for text in texts:
                    ids = self.sp.encode(text, out_type=int)
                    if len(ids) > self.text_len: ids = ids[:self.text_len]
                    else: ids = ids + [self.sp.pad_id()] * (self.text_len - len(ids))
                    ids_list.append(ids)
                ids_tensor = torch.tensor(ids_list, dtype=torch.long, device=device)
                mask = (ids_tensor != self.sp.pad_id()).long()
                seq_lens = mask.sum(dim=1).long()
                context = self.model(ids_tensor, mask)
                return [u[:v] for u, v in zip(context, seq_lens)]
        
        wrapper.text_encoder = T5EncoderWrapper(t5_model, sp)
        
        # WanModel Loading
        from ..dreamidv_wan.modules.attention import set_attention_mode
        set_attention_mode(attention_mode)
        logging.info(f"[DreamID-V Wrapper] Loading WanModel ({backend})...")
        
        wrapper.model = ModelClass(
            model_type=cfg.model_type,
            dim=cfg.dim,
            ffn_dim=cfg.ffn_dim,
            freq_dim=cfg.freq_dim,
            in_dim=cfg.in_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            window_size=cfg.window_size,
            qk_norm=cfg.qk_norm,
            cross_attn_norm=cfg.cross_attn_norm,
            eps=cfg.eps
        )
        wrapper.model.to(dtype=base_dtype)
        
        params_fp32 = ["patch_embedding", "motion_encoder", "condition_embedding", "norm", "bias", "time_in", "time_", "img_emb", "modulation", "text_embedding", "adapter", "add", "ref_conv", "audio_proj"]
        for name, param in wrapper.model.named_parameters():
             if any(k in name for k in params_fp32): param.data = param.data.to(dtype=torch.float32)

        state = torch.load(dreamidv_path, map_location='cpu')
        sd = wrapper.model.state_dict()
        new_state = {}
        for k, v in state.items():
            if k in sd: new_state[k] = v.to(dtype=sd[k].dtype)
            else: new_state[k] = v
        wrapper.model.load_state_dict(new_state, strict=False)
        wrapper.model.eval().requires_grad_(False).to(device)
        
        if backend == "faster":
            wrapper.load_image_latent_ref_ip_video = faster_load_data_tensor.__get__(wrapper, DreamIDV_Faster)
            wrapper.generate = faster_generate_tensor.__get__(wrapper, DreamIDV_Faster)
            wrapper.is_faster = True
        else:
            from ..dreamidv_wan.wan_swapface import DreamIDV
            wrapper.load_image_latent_ref_ip_video = DreamIDV.load_image_latent_ref_ip_video.__get__(wrapper, DreamIDVWrapper)
            wrapper.generate = DreamIDV.generate.__get__(wrapper, DreamIDVWrapper)
            wrapper.is_faster = False
        
        DreamIDV_ModelLoader_Wrapper_TTP._cached_wrapper = wrapper
        DreamIDV_ModelLoader_Wrapper_TTP._cached_paths = current_paths
        
        return (wrapper,)

NODE_CLASS_MAPPINGS = {
    "DreamIDV_ModelLoader_Wrapper_TTP": DreamIDV_ModelLoader_Wrapper_TTP
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamIDV_ModelLoader_Wrapper_TTP": "DreamID-V Model Loader (Wrapper)"
}