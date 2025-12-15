import os
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import MaskablePPO  # 如果没有安装 sb3_contrib，可以用普通的 PPO
import torch

# ================= 🚀 配置区域 =================
# 关键点：开启 use_lidar=True，这是拿高分的核心
ENV_ID = "FlappyBird-v0"
MODEL_DIR = "trained_models"
LOG_DIR = "logs"
MODEL_NAME = "FlappyBird_Master"
TOTAL_TIMESTEPS = 1_000_000  # 建议训练 50万-100万步

# 创建目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env(render_mode=None):
    """
    创建环境的辅助函数
    use_lidar=True: 开启激光雷达，大大降低训练难度
    """

    def _init():
        env = gym.make(
            ENV_ID,
            render_mode=render_mode,
            use_lidar=True,  # <--- 核心：开启雷达
            background=None  # 关闭背景以加速训练(可选)
        )
        return env

    return _init


def train():
    print("🚀 开始训练大师级模型...")
    print(f"📌 设备: {'GPU (cuda)' if torch.cuda.is_available() else 'CPU'}")

    # 使用多进程环境加速训练 (4个进程)
    # 如果报错，可以改回 DummyVecEnv([make_env()])
    num_cpu = 4
    env = SubprocVecEnv([make_env() for _ in range(num_cpu)])

    # 定义 PPO 模型参数 (经过优化的参数)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,  # 稍微降低学习率
        n_steps=2048,
        batch_size=64,
        n_epochs=10,  # 每次更新多学几遍
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # 熵系数，鼓励探索
        tensorboard_log=LOG_DIR,  # 可视化日志
        device="auto"
    )

    # 自动保存回调：每 10万步保存一次模型
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // num_cpu,
        save_path=MODEL_DIR,
        name_prefix="ppo_flappy"
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, progress_bar=True)
        # 保存最终模型
        final_path = os.path.join(MODEL_DIR, MODEL_NAME)
        model.save(final_path)
        print(f"✅ 训练完成！模型已保存至: {final_path}")
    except KeyboardInterrupt:
        print("⚠️ 训练手动停止，正在保存当前模型...")
        model.save(os.path.join(MODEL_DIR, "interrupted_model"))
    finally:
        env.close()


def test():
    print("👀 正在加载大师级模型进行演示...")

    model_path = os.path.join(MODEL_DIR, MODEL_NAME + ".zip")
    if not os.path.exists(model_path):
        # 尝试寻找中间保存的模型
        print(f"❌ 找不到 {model_path}，请检查是否训练完成。")
        return

    # 测试时开启渲染，并且必须保持 use_lidar=True
    env = gym.make(ENV_ID, render_mode="human", use_lidar=True)

    model = PPO.load(model_path, env=env)

    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        print(f"🎬 Episode {ep + 1} 开始...")
        while not done:
            # deterministic=True 让动作更稳定
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            if truncated:
                pass
            total_reward += reward

            # 如果分数太高，不想看了可以按 Ctrl+C
            if info['score'] > 1000:
                print("✨ 分数超过1000，太强了，自动跳过...")
                # break

        print(f"🏁 Episode {ep + 1} 得分: {info['score']}")

    env.close()


if __name__ == "__main__":
    # 1. 先运行 train()
    # 2. 训练完后注释掉 train()，运行 test()

    # train()
    test()
