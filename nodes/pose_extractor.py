"""
DreamID-V Pose Extractor Node (DWPose + Offload Support)
"""
import torch
import logging
import os
import gc

os.environ["GLOG_minloglevel"] = "2"

class DreamIDV_PoseExtractor_TTP:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_video": ("IMAGE",),
                "reference_image": ("IMAGE",), # DWPose 实际上不用这个，但为了兼容性保留
                "pose_backend": (["dwpose", "mediapipe"], {"default": "dwpose"}),
                "max_frames": ("INT", {"default": 81, "min": 5, "max": 999, "step": 4}),
                "static_mode": ("BOOLEAN", {"default": True}),
                "min_detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                # 新增 Offload 选项
                "offload_pose_models": ("BOOLEAN", {"default": True, "tooltip": "Unload DWPose models from VRAM after extraction"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("pose_video", "mask_video", "actual_frames")
    FUNCTION = "extract_pose"
    CATEGORY = "DreamID-V_TTP"
    
    def extract_pose(self, reference_video, reference_image, pose_backend, max_frames, static_mode, min_detection_confidence, offload_pose_models):
        
        # 1. 帧数对齐
        adjusted_frames = ((max_frames - 1) // 4) * 4 + 1
        video_tensor = reference_video[:adjusted_frames]
        actual_frames = video_tensor.shape[0]

        try:
            if pose_backend == "dwpose":
                logging.info("[PoseExtractor] Backend: DWPose")
                # 延迟加载，防止没文件报错
                from ..utils.dwpose_handler import DWPoseDetector
                
                # 初始化 (加载模型到显存)
                detector = DWPoseDetector()
                
                # 推理
                pose_tensor, mask_tensor = detector(video_tensor)
                
                # 显存释放逻辑
                if offload_pose_models:
                    detector.release()
                    del detector
                    
                logging.info(f"[PoseExtractor] ✓ Done ({actual_frames} frames)")
                return (pose_tensor, mask_tensor, actual_frames)
                
            else:
                # MediaPipe Fallback (旧版逻辑)
                logging.info("[PoseExtractor] Backend: MediaPipe")
                from ..utils.mediapipe_utils import generate_pose_video, generate_mask_video
                
                video_np = video_tensor.cpu().numpy()
                ref_image_np = reference_image[0].cpu().numpy()
                
                pose_np = generate_pose_video(video_np, ref_image_np, adjusted_frames, static_mode=static_mode, min_detection_confidence=min_detection_confidence)
                mask_np = generate_mask_video(video_np, ref_image_np, adjusted_frames, static_mode=static_mode, min_detection_confidence=min_detection_confidence)
                
                return (torch.from_numpy(pose_np).float(), torch.from_numpy(mask_np).float(), actual_frames)
            
        except Exception as e:
            logging.error(f"[PoseExtractor] Error: {e}")
            raise e

NODE_CLASS_MAPPINGS = {
    "DreamIDV_PoseExtractor_TTP": DreamIDV_PoseExtractor_TTP
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamIDV_PoseExtractor_TTP": "DreamID-V Pose Extractor (DWPose+Offload)"
}