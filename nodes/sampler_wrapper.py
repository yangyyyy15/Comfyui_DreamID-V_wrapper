"""
DreamID-V Sampler Wrapper
Uses DreamIDV.generate() method
Updated to support 'Faster' backend
"""

import torch
import numpy as np
import logging
import os
import cv2
from PIL import Image

import folder_paths

# Import from local copy  
from ..dreamidv_wan.configs import SIZE_CONFIGS


# Wan2.1 Latent Mean/Std
WAN21_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 
              0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
WAN21_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 
             3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]

class ComfyUIVAEAdapter:
    """
    Adapter to wrap ComfyUI VAE with DreamID-V compatible interface.
    """
    
    def __init__(self, comfy_vae, device):
        self.comfy_vae = comfy_vae
        self.device = device
        self.z_dim = 16 
        self.mean = torch.tensor(WAN21_MEAN, device=device).view(1, 16, 1, 1, 1) 
        self.std = torch.tensor(WAN21_STD, device=device).view(1, 16, 1, 1, 1)
        self.model = self
    
    def encode(self, videos, device):
        results = []
        for video in videos:
            # video is [C, T, H, W] in [-1, 1]
            video = video.to(device)
            # [C, T, H, W] -> [T, H, W, C] in [0, 1]
            video_bhwc = video.permute(1, 0, 2, 3).permute(0, 2, 3, 1)
            video_bhwc = (video_bhwc + 1.0) / 2.0
            latent = self.comfy_vae.encode(video_bhwc)
            if latent.dim() == 4: latent = latent.unsqueeze(0)
            
            if self.mean.device != latent.device:
                self.mean = self.mean.to(latent.device)
                self.std = self.std.to(latent.device)
            
            latent = (latent - self.mean) / self.std
            latent = latent.squeeze(0)
            results.append(latent)
        return results
    
    def decode(self, zs):
        results = []
        for z in zs:
            z_batch = z.unsqueeze(0)
            if self.mean.device != z_batch.device:
                self.mean = self.mean.to(z_batch.device)
                self.std = self.std.to(z_batch.device)
            z_raw = z_batch * self.std + self.mean
            decoded = self.comfy_vae.decode(z_raw)
            decoded = decoded * 2.0 - 1.0
            decoded = decoded.permute(0, 3, 1, 2).permute(1, 0, 2, 3)
            results.append(decoded.clamp(-1, 1))
        return results


class DreamIDV_Sampler_Wrapper_TTP:
    """
    Sampler using DreamIDV.generate()
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dreamidv_wrapper": ("DREAMIDV_WRAPPER",),
                "vae": ("VAE",),
                "ref_video": ("IMAGE",),
                "ref_image": ("IMAGE",),  
                "mask_video": ("IMAGE",),
                "size": (["832*480", "1280*720", "480*832", "720*1280", "custom"], {"default": "1280*720"}),
                "frame_num": ("INT", {"default": 81, "min": 1, "max": 200, "step": 4}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "offload_t5": ("BOOLEAN", {"default": True, "tooltip": "Offload T5 to CPU after text encoding"}),
                "offload_video_model": ("BOOLEAN", {"default": True, "tooltip": "Offload Video Diffusion Model to CPU"}),
            },
            "optional": {
                "pose_video": ("IMAGE",), # Optional for Faster
                "custom_width": ("INT", {"default": 1280, "min": 64, "max": 2048, "step": 8}),
                "custom_height": ("INT", {"default": 720, "min": 64, "max": 2048, "step": 8}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "sample"
    CATEGORY = "DreamID-V/Wrapper"
    
    def sample(self, dreamidv_wrapper, vae, ref_video, ref_image, mask_video,
               size, frame_num, steps, cfg_scale, shift, seed, offload_t5, offload_video_model,
               pose_video=None, custom_width=1280, custom_height=720):
        
        # Check backend flag injected by loader
        is_faster = getattr(dreamidv_wrapper, "is_faster", False)
        
        logging.info(f"[DreamID-V Wrapper] Starting generation (Backend: {'FASTER' if is_faster else 'ORIGINAL'})...")
        logging.info(f"  Frames: {frame_num}, Steps: {steps}, CFG: {cfg_scale}, Shift: {shift}")
        
        if is_faster:
            # Faster config: 3 inputs [Video, Mask, Image]
            logging.info("[DreamID-V Wrapper] Config: Faster (Pose input ignored)")
            ref_paths = [
                ref_video,
                mask_video,
                ref_image
            ]
            if steps > 25:
                logging.warning(f"[DreamID-V Wrapper] Note: 'Faster' model typically uses fewer steps (e.g. 12-15). Current steps: {steps}")
                
        else:
            # Original config: 4 inputs [Video, Mask, Image, Pose]
            if pose_video is None:
                raise ValueError("[DreamID-V Wrapper] Original backend requires 'pose_video' input! Please connect a Pose video.")
            logging.info("[DreamID-V Wrapper] Config: Original (Pose input used)")
            ref_paths = [
                ref_video,
                mask_video,
                ref_image,
                pose_video
            ]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Attach VAE Adapter
        pipeline = dreamidv_wrapper
        pipeline.vae = ComfyUIVAEAdapter(vae, device)
        
        # Determine Size
        if size == 'custom':
            size_tuple = (custom_width, custom_height)
        else:
            size_tuple = SIZE_CONFIGS[size]
        
        # Generate
        # Note: ref_paths are now Tensors, supported by our monkey-patched loader method
        generated = pipeline.generate(
            input_prompt='change face',
            paths=ref_paths,
            size=size_tuple,
            frame_num=frame_num,
            shift=shift,
            sample_solver='unipc', # Faster only supports unipc
            sampling_steps=steps,
            guide_scale_img=cfg_scale,
            seed=seed,
            offload_t5=offload_t5,
            offload_video_model=offload_video_model,
            return_latent=True
        )
        
        # Handle Output
        logging.info(f"[DreamID-V Wrapper] Generated latent: {generated.shape}")
        
        # Convert to ComfyUI LATENT [1, C, T, H, W]
        latent = generated.unsqueeze(0)
        
        # Unshift Latent for ComfyUI VAE (which expects Raw Latent)
        device = latent.device
        mean = torch.tensor(WAN21_MEAN, device=device).view(1, 16, 1, 1, 1)
        std = torch.tensor(WAN21_STD, device=device).view(1, 16, 1, 1, 1)
        
        latent = latent * std + mean
        
        logging.info(f"[DreamID-V Wrapper] ✓ Generation Complete")
        
        return ({"samples": latent},)


NODE_CLASS_MAPPINGS = {
    "DreamIDV_Sampler_Wrapper_TTP": DreamIDV_Sampler_Wrapper_TTP
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamIDV_Sampler_Wrapper_TTP": "DreamID-V Sampler (Wrapper)"
}