import os
import glob
import numpy as np
import random
import torch
import json
from main.inference.motionctrl_cmcm_evaluate import run_motionctrl_inference

def parse_trajectory_file(txt_path, device=torch.device('cuda')):
    """
    针对RealEstate10K的相机轨迹
    解析单个相机轨迹 .txt 文件，仅获取相机外参矩阵。
    
    参数:
        txt_path (str): 相机轨迹文件路径。
        device (torch.device): 张量设备（'cpu' 或 'cuda'）。
    
    返回:
        video_url (str): 视频URL/ID，用于获取原始视频的路径。
        c2ws (torch.Tensor): 相机外参矩阵 [N, 3, 4]
    """
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 第1行是视频URL
    video_url = lines[0]

    # 解析相机外参
    w2cs = []
    for frame_line in lines[1:]:
        cols = frame_line.split()
        if len(cols) >= 19:  # 基础检查：确保至少有19列
            ext_cols = list(map(float, cols[7:19]))  # 提取外参（7-19列）
            ext_matrix = np.array(ext_cols).reshape(3, 4)
            w2cs.append(torch.tensor(ext_matrix, dtype=torch.float).to(device))

    # 转换为Tensor并计算c2w
    w2cs = torch.stack(w2cs, dim=0)  # [N, 3, 4]
    bottom = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float).to(device)
    bottom = bottom.repeat(w2cs.shape[0], 1, 1)  # [N, 1, 4]
    w2cs_4x4 = torch.cat((w2cs, bottom), dim=1)  # [N, 4, 4]
    c2ws_4x4 = torch.linalg.inv(w2cs_4x4)        # [N, 4, 4]
    c2ws = c2ws_4x4[:, :3, :]                   # [N, 3, 4]

    return video_url, c2ws

def sample_frames(frame_folder, num_frames=25):
    """
    针对RealEstate10K
    从帧文件夹中均匀或随机采样帧图像。
    
    参数:
        frame_folder (str): 帧文件夹路径。
        num_frames (int): 采样的帧数。
        
    返回:
        frame_paths (list[str]): 采样的帧路径。
    """
    if not os.path.exists(frame_folder):
        return []
    
    # 获取所有帧文件
    all_frames = sorted(glob.glob(os.path.join(frame_folder, "*.png")))
    if len(all_frames) == 0:
        return []
    
    # 如果帧数不足25，直接返回所有帧
    if len(all_frames) <= num_frames:
        return all_frames
    
    # 均匀采样25帧
    indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
    sampled_frames = [all_frames[i] for i in indices]
    return sampled_frames


def load(
    camera_root="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/zhengsixiao/RealEstate10K_camera/test",
    dataset_root="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/zhengsixiao/RealEstate10K/dataset/test",
    video_root="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/zhengsixiao/RealEstate10K/videos/",
    num_frames=25
):
    """
    1. 遍历 RealEstate10K_camera/test 下所有 .txt 文件
    2. 解析相机轨迹，获取 extrinsics
    3. 从 dataset/test 下对应帧文件夹采样 num_frames 帧
    4. 定位到视频文件
    """
    # 获取所有轨迹文件
    txt_files = glob.glob(os.path.join(camera_root, "*.txt"))
    
    results = []  # 存放解析结果
    
    cnt=0
    for txt_path in txt_files:
        if cnt > 3:
            break
        cnt=cnt+1
        # 获取视频ID（基于 txt 文件名）
        basename = os.path.splitext(os.path.basename(txt_path))[0]  # e.g., "2bec33eeeab0bb9d"

        # 解析轨迹
        video_url, extrinsics = parse_trajectory_file(txt_path)

        # 帧文件夹路径
        frame_folder = os.path.join(dataset_root, basename)
        frame_paths = sample_frames(frame_folder, num_frames=num_frames)

        # 原始视频路径
        video_file = os.path.join(video_root, f"{basename}.mp4")
        if not os.path.exists(video_file):
            video_file = None

        # 组装结果
        info_dict = {
            "txt_file": txt_path,
            "video_id": basename,
            "video_url": video_url,
            "video_file": video_file,
            "extrinsics": extrinsics,  # [N, 3, 4]
            "frame_paths": frame_paths  # 采样的25帧图像路径
        }

        results.append(info_dict)

    return results

def save_extrinsics_to_json(parsed_results, output_json_root="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/project/MotionCtrl/examples/camera_poses_evaluate"):
    """
    将从 load() 获取的相机外参（extrinsics）保存到指定 JSON 文件夹中
    将RealEstate10K 轨迹格式转为motionctrl的轨迹格式
    
    参数:
        parsed_results (list): 从 load() 获取的解析结果列表。
        output_json_root (str): 保存 JSON 文件的根目录。
    """
    os.makedirs(output_json_root, exist_ok=True)

    for item in parsed_results:
        video_id = item["video_id"]
        extrinsics = item["extrinsics"]  # [N, 3, 4]
        json_path = os.path.join(output_json_root, f"{video_id}_pose.json")

        # 转换为 [N, 12] 格式并保存
        extrinsics_flat = extrinsics.reshape(-1, 12).cpu().tolist()
        with open(json_path, 'w') as f:
            json.dump(extrinsics_flat, f, indent=4)
        
        print(f"Saved extrinsics for {video_id} to {json_path}")


if __name__ == "__main__":
    
    # 获取测试数据集信息
    parsed_results = load()
    print("loaded")
    
    # 保存相机外参到 JSON 文件(转成motionctrl格式)
    output_pose_dir="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/project/MotionCtrl/examples/camera_poses_evaluate"
    save_extrinsics_to_json(parsed_results, output_json_root=output_pose_dir)

    # 打印前几个结果进行检查
    for item in parsed_results[:3]:
        print("轨迹文件:", item["txt_file"])
        print("视频ID:", item["video_id"])
        print("视频URL:", item["video_url"])
        print("原始视频文件:", item["video_file"])
        print("外参矩阵形状:", item["extrinsics"].shape)
        print("采样的帧图像路径 (前5帧):", item["frame_paths"][:5])
        # print(len(item["frame_paths"]))
        print("-------------------------------------------------\n")
    
    # 遍历 parsed_results，调用 run_motionctrl_inference（运行motionctrl测试）
    for idx, item in enumerate(parsed_results):
        video_id = item["video_id"]
        frame_paths = item["frame_paths"]
        extrinsics = item["extrinsics"]

        if not frame_paths:
            print(f"[{idx}] {video_id} has no frames, skip.")
            continue
        
        # 取第 1 帧作为 input
        image_input = frame_paths[0]
        
        # 根据video_id 获取对应的 .json 文件
        pose_path = os.path.join(output_pose_dir, f"{video_id}_pose.json")
        print(f"Pose path: {pose_path}")
        
        # 读取，查看shape
        import cv2
        image = cv2.imread(image_input)
        print(f"Image shape: {image.shape}")

        
        # 你可以根据 extrinsics 的 shape 等信息做一些检查，这里略

        print(f"\n=== [{idx}] Start inference for video_id={video_id} ===")
        print(f"Use first frame: {image_input}")

        # 调用推理函数
        # 注意：下面这些参数可以根据你的实际需求修改
        run_motionctrl_inference(
            seed=12345,
            ckpt="../../model/motionctrl/motionctrl_svd.ckpt",
            config="configs/inference/config_motionctrl_cmcm.yaml",
            savedir=f"outputs/motionctrl_svd/{video_id}",  # 每个视频ID单独输出
            savefps=10,
            ddim_steps=25,
            frames=14,  # 需要与 .json 内外参帧数匹配
            image_input=image_input,
            fps=10,
            motion=127,
            cond_aug=0.02,
            decoding_t=1,
            resize=True,
            height=576,
            width=1024,
            sample_num=1,
            transform=True,
            # 这里最关键：告诉脚本去读我们刚才保存的 JSON 文件夹
            pose_dir=pose_path,
            speed=2.0,
            save_images=False,
            device="cuda"
        )    
    