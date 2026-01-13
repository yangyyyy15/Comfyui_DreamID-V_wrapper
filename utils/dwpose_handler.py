import os
import torch
import numpy as np
import cv2
import urllib.request
import shutil
import gc
import folder_paths
from tqdm import tqdm

# 导入你复制过来的库
from .dwpose import util as dwpose_util
from .dwpose.wholebody import Wholebody

# DWPose 模型配置
DWPOSE_FILES = {
    "dw-ll_ucoco_384.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
    "yolox_l.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
}

def find_or_download_model(filename, url):
    """智能路径查找：优先 models/dwpose，其次兼容 ControlNet 路径"""
    target_dir = os.path.join(folder_paths.models_dir, "dwpose")
    
    # 搜索列表
    possible_dirs = [
        target_dir,
        os.path.join(folder_paths.models_dir, "Annotators"),
        os.path.join(folder_paths.models_dir, "Annotators", "dwpose"),
        # 旧版兼容
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "comfyui_controlnet_aux", "ckpts", "lllyasviel", "Annotators"),
    ]

    for d in possible_dirs:
        if os.path.exists(d):
            path = os.path.join(d, filename)
            if os.path.exists(path):
                print(f"[DreamID-V Wrapper] Found DWPose model at: {path}")
                return path

    # 下载逻辑
    os.makedirs(target_dir, exist_ok=True)
    dst_path = os.path.join(target_dir, filename)
    print(f"[DreamID-V Wrapper] Downloading {filename} to {target_dir}...")
    try:
        with urllib.request.urlopen(url) as r, open(dst_path, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:
        if os.path.exists(dst_path): os.remove(dst_path)
        raise e
    return dst_path

def ensure_dwpose_models():
    yolox = find_or_download_model("yolox_l.onnx", DWPOSE_FILES["yolox_l.onnx"])
    dwpose = find_or_download_model("dw-ll_ucoco_384.onnx", DWPOSE_FILES["dw-ll_ucoco_384.onnx"])
    return yolox, dwpose

# --- 绘图工具 ---
def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    # 调用 utils 里的绘图逻辑
    canvas = dwpose_util.draw_bodypose(canvas, candidate, subset)
    canvas = dwpose_util.draw_handpose(canvas, hands)
    canvas = dwpose_util.draw_facepose(canvas, faces)
    return canvas

def draw_face_mask_from_points(faces, H, W):
    mask = np.zeros((H, W, 3), dtype=np.uint8)
    max_area = 0
    best_hull = None
    for face in faces:
        valid_points = []
        for pt in face:
            if pt[0] >= 0 and pt[1] >= 0:
                valid_points.append([int(pt[0] * W), int(pt[1] * H)])
        if len(valid_points) > 3:
            pts = np.array(valid_points, np.int32)
            hull = cv2.convexHull(pts.reshape((-1, 1, 2)))
            if cv2.contourArea(hull) > max_area:
                max_area = cv2.contourArea(hull)
                best_hull = hull
    if best_hull is not None:
        cv2.fillPoly(mask, [best_hull], (255, 255, 255))
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
    return mask

# --- 主处理器类 ---
class DWPoseDetector:
    def __init__(self):
        det_path, pose_path = ensure_dwpose_models()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DreamID-V Wrapper] Initializing DWPose on {device}...")
        self.pose_estimation = Wholebody(det_path, pose_path, device=device)

    def __call__(self, frames_tensor):
        frames_np = (frames_tensor.cpu().numpy() * 255).astype(np.uint8)
        pose_res, mask_res = [], []
        
        for frame in tqdm(frames_np, desc="DWPose Extracting"):
            H, W, _ = frame.shape
            
            # 1. 推理
            candidate, subset, _ = self.pose_estimation(frame)
            nums, keys, locs = candidate.shape
            
            # 2. 坐标归一化
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            
            # 3. [关键修复] 数据重塑 (Reshape)
            # DWPose utils 需要 body 是 (N*18, 2) 的格式，而不是 (N, 18, 2)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            
            # 4. 过滤低分点
            score = subset[:, :18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3: 
                        score[i][j] = int(18 * i + j)
                    else: 
                        score[i][j] = -1
            
            # 5. 处理可见性
            un_visible = subset < 0.3
            candidate[un_visible] = -1

            faces = candidate[:, 24:92]
            hands = np.vstack([candidate[:, 92:113], candidate[:, 113:]])
            
            # 6. 构造数据包
            bodies = dict(candidate=body, subset=score)
            
            # 7. 绘图
            pose_res.append(draw_pose(dict(bodies=bodies, hands=hands, faces=faces), H, W))
            mask_res.append(draw_face_mask_from_points(faces, H, W))
            
        return (torch.from_numpy(np.array(pose_res)).float() / 255.0, 
                torch.from_numpy(np.array(mask_res)).float() / 255.0)

    def release(self):
        """显存释放逻辑"""
        print("[DreamID-V Wrapper] Offloading DWPose models...")
        if hasattr(self, 'pose_estimation'):
            del self.pose_estimation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()