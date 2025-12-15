import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import DQN  # <--- 1. 改用 DQN
from huggingface_hub import hf_hub_download
import os

def load_and_play_best_model():
    print(">>> 正在从 Hugging Face 下载可用模型 (nsanghi/dqn-flappy-sb3)...")

    # 2. 修改下载参数
    # 使用现存的公开仓库，并设置 token=False 避免 401 报错
    try:
        model_path = hf_hub_download(
            repo_id="nsanghi/dqn-flappy-sb3",
            filename="dqn-flappy-sb3.zip",
            token=False  # <--- 关键：强制匿名下载，解决 401 认证错误
        )
    except Exception as e:
        print(f"下载出错: {e}")
        return

    print(f">>> 模型已下载至: {model_path}")

    # 3. 创建环境
    # 注意：大多数 DQN 模型训练时 render_mode="human"
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)

    # 4. 加载模型
    # 改用 DQN.load，并传入 env.observation_space 以防版本不匹配
    model = DQN.load(model_path, custom_objects={"observation_space": env.observation_space})

    # 5. 开始演示
    print(">>> 开始演示 (按 Ctrl+C 退出)...")
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    load_and_play_best_model()