import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import DQN  # 注意这里改成了 DQN
from huggingface_hub import hf_hub_download
import os


def load_and_play_champion():
    print(">>> 🚀 正在下载无敌版模型 (DQN)...")

    # 1. 下载模型文件
    # 使用 nsanghi/dqn-flappy-sb3，这是一个公认的高分模型
    # token=False 强制不使用本地过期的密钥，解决 401 错误
    try:
        model_path = hf_hub_download(
            repo_id="nsanghi/dqn-flappy-sb3",
            filename="dqn-flappy-sb3.zip",
            token=False  # <--- 关键修复：强制匿名下载
        )
    except Exception as e:
        print(f"自动下载失败: {e}")
        print("💡 备用方案: 请手动下载文件 put in project folder.")
        print("下载地址: https://huggingface.co/nsanghi/dqn-flappy-sb3/resolve/main/dqn-flappy-sb3.zip")
        return

    print(f">>> ✅ 模型已就绪: {model_path}")

    # 2. 创建环境
    # 注意：这个模型训练时 render_mode 是 human
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)

    # 3. 加载模型
    # 以此模型为例，它是用 DQN 训练的，所以必须用 DQN.load
    model = DQN.load(model_path, custom_objects={"observation_space": env.observation_space})

    # 4. 开始演示
    print(">>> 🎮 开始演示 (按 Ctrl+C 停止)...")
    obs, _ = env.reset()
    total_score = 0

    while True:
        # DQN 的预测逻辑和 PPO 一样
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # 实时显示分数
        current_score = info.get('score', 0)
        if current_score > total_score:
            total_score = current_score
            print(f"\r当前分数: {total_score}", end="")

        if terminated or truncated:
            print(f"\n💀 游戏结束! 最终得分: {total_score}")
            obs, _ = env.reset()
            total_score = 0


if __name__ == "__main__":
    load_and_play_champion()