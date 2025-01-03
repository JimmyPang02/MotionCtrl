import os
import csv
import numpy as np
import json

# 假设视频生成模型的函数定义如下
def generate_video_from_caption_and_trajectory(caption, trajectory):
    # 这里是你实际的视频生成模型的代码
    # 返回生成的视频
    generated_video = f"generated_video_{caption}.mp4"
    return generated_video

# 读取CSV文件，获取视频信息
def read_video_info(csv_file):
    video_info = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            videoid = row['videoid']
            page_dir = row['page_dir']
            name = row['name']
            if page_dir not in video_info:
                video_info[page_dir] = {}
            video_info[page_dir][videoid] = name
    return video_info

# 遍历所有视频，生成并保存轨迹和caption
def process_videos(root_dir, video_info):
    trajs = []
    traj_prompt = []
    total_videos = sum(len(os.listdir(os.path.join(root_dir, d))) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
    processed_videos = 0

    for page_dir in os.listdir(root_dir):
        page_dir_path = os.path.join(root_dir, page_dir)
        if os.path.isdir(page_dir_path):
            for videoid_dir in os.listdir(page_dir_path):
                # 提取videoid
                videoid = videoid_dir.split('_')[0]
                videoid_path = os.path.join(page_dir_path, videoid_dir)
                if os.path.isdir(videoid_path):
                    # 获取轨迹文件
                    trajectory_file = os.path.join(videoid_path, 'sparse_gs.npy')
                    if os.path.exists(trajectory_file):
                        # 获取caption
                        caption = video_info.get(page_dir, {}).get(videoid)
                        if caption:
                            # 保存轨迹文件路径和caption
                            trajs.append(trajectory_file)
                            traj_prompt.append(caption)
                        else:
                            print(f"No caption found for videoid: {videoid} in page_dir: {page_dir}")
                    else:
                        print(f"Trajectory file not found for videoid: {videoid} in page_dir: {page_dir}")

                    processed_videos += 1
                    # print(f"Processed {processed_videos}/{total_videos} videos")

    # 保存到字典中
    omom_prompt_traj = {
        'prompts': traj_prompt,
        'trajs': trajs
    }

    # 保存到指定路径
    save_path = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/data/webvid10m/omom_prompt_traj.json'
    with open(save_path, 'w') as f:
        json.dump(omom_prompt_traj, f)

    print(f"Data saved to {save_path}")

# 主函数
def main():
    root_dir = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/data/webvid10m/val_512_32'  # 数据集根目录
    csv_file = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/data/webvid10m/0000.csv'  # CSV文件路径
    video_info = read_video_info(csv_file)
    process_videos(root_dir, video_info)

if __name__ == "__main__":
    main()