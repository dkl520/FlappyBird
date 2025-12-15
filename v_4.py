import gymnasium as gym
import flappy_bird_gymnasium
import torch
import os
import numpy as np
from stable_baselines3 import PPO  # 🔥 核心改变：使用 PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# ==========================================
# ⚙️ 核心配置 (State of the Art)
# ==========================================
MODELS_DIR = "models/flappy_ppo_v3"
LOG_DIR = "logs/flappy_ppo_v3"
MODEL_NAME = "flappy_bird_ppo_best"

# ⚡ 并行环境数：CPU核心数越多越好，通常设置 4-8
N_ENVS = 4

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ==========================================
# 🚀 训练环境构建
# ==========================================
def make_env():
    # 🔥 关键秘籍：use_lidar=True
    # 这会给 AI 提供 180 个距离传感器数据，相比只有坐标，这简直是开了挂
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=True)
    return env


def train():
    print(f">>> [初始化] 启动 {N_ENVS} 个并行环境 (PPO算法)...")

    # 创建并行环境
    env = make_vec_env(make_env, n_envs=N_ENVS, monitor_dir=LOG_DIR)

    # PPO 超参数 (针对 Flappy Bird 调优)
    # 相比 DQN，PPO 对超参数不那么敏感，更容易训练
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,  # 标准学习率
        n_steps=2048,  # 每次更新的采样步数
        batch_size=64,  # 小批量大小
        n_epochs=10,  # 每次更新的迭代次数
        gamma=0.99,  # 折扣因子
        gae_lambda=0.95,  # 优势估计参数
        clip_range=0.2,  # PPO 裁剪范围
        ent_coef=0.01,  # 🔥 熵系数：强制 AI 尝试不同动作，防止早熟
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # 网络结构
            activation_fn=torch.nn.Tanh
        ),
    )

    # 检查是否有已存在的模型继续训练
    final_path = f"{MODELS_DIR}/{MODEL_NAME}.zip"
    if os.path.exists(final_path):
        print(f">>> ♻️ 检测到旧模型，正在加载继续训练...")
        model = PPO.load(final_path, env=env)

    # 回调函数：保存最优模型
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=LOG_DIR,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # 训练步数：PPO 效率很高，50万步通常就能达到“不死”状态
    # 也就是现实时间大约 5-10 分钟
    TOTAL_STEPS = 1_000_000
    print(f">>> [开始] 目标训练步数: {TOTAL_STEPS}...")

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=eval_callback,
            progress_bar=True
        )
        model.save(f"{MODELS_DIR}/{MODEL_NAME}")
        print(">>> ✅ 训练完成！")
    except KeyboardInterrupt:
        print(">>> 🛑 训练中断，正在保存当前模型...")
        model.save(f"{MODELS_DIR}/interrupted_ppo")

    env.close()


# ==========================================
# 🧪 测试函数 (享受成果)
# ==========================================
def test():
    model_path = f"{MODELS_DIR}/best_model.zip"
    if not os.path.exists(model_path):
        model_path = f"{MODELS_DIR}/{MODEL_NAME}.zip"
        if not os.path.exists(model_path):
            print("❌ 没有找到模型文件！请先运行训练。")
            return

    print(f">>> 🎮 正在加载模型: {model_path}")

    # 测试时开启 render_mode="human" 观看 AI 操作
    # 记得也要开启 use_lidar=True，否则输入维度对不上
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)

    model = PPO.load(model_path)

    episodes = 10
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            # deterministic=True 让 AI 发挥稳定实力，不进行随机探索
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score = info.get('score', score)

        print(f"第 {ep + 1} 局得分: {score}")

    env.close()


if __name__ == "__main__":
    # 1. 先运行训练 (只需运行一次，大约 5-10 分钟)
    # train()

    # 2. 训练完成后注释掉 train()，取消下面 test() 的注释来观看
    test()