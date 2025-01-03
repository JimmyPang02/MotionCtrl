import json

# 读取JSON文件
def load_omom_prompt_traj(json_file):
    with open(json_file, 'r') as f:
        omom_prompt_traj = json.load(f)
    return omom_prompt_traj

# 检查prompts和trajs的长度
def check_lengths(omom_prompt_traj):
    prompts = omom_prompt_traj['prompts']
    trajs = omom_prompt_traj['trajs']
    
    print(f"Length of prompts: {len(prompts)}")
    print(f"Length of trajs: {len(trajs)}")
    
    if len(prompts) == len(trajs):
        print("The lengths of prompts and trajs are the same.")
    else:
        print("The lengths of prompts and trajs are different.")

# 主函数
def main():
    json_file = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/data/webvid10m/omom_prompt_traj.json'  # JSON文件路径
    omom_prompt_traj = load_omom_prompt_traj(json_file)
    check_lengths(omom_prompt_traj)

if __name__ == "__main__":
    main()