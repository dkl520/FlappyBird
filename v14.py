import os
import time
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# ================= 🚀 超参数配置 =================
ENV_ID = "FlappyBird-v0"
MODEL_DIR = "manual_models"
MODEL_NAME = "ppo_flappy_final14"
TOTAL_TIMESTEPS = 500_000
DEVICE = "cpu"


# ================= 🏃 训练流程 =================
def train():
    print("🚀 准备开始训练 (Stable-Baselines3 版)...")

    # 1. 创建环境
    env = gym.make(ENV_ID, use_lidar=True, background=None)
    env = Monitor(env)

    # 2. 路径处理
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, "checkpoints")  # 定义子文件夹
    os.makedirs(checkpoint_path, exist_ok=True)  # 🟢 修复：创建 checkpoints 文件夹

    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    # 3. 断点续训逻辑
    if os.path.exists(f"{model_path}.zip"):
        print(f"🔄 发现上次训练模型: {model_path}.zip")
        try:
            model = PPO.load(model_path, env=env, device=DEVICE)
            print("✅ 模型加载成功！将在该模型基础上继续训练 (Resume Training)")
        except ValueError as e:
            print(f"❌ 模型加载失败 (可能是网络结构或Lidar参数改变导致): {e}")
            print("⚠️ 请删除旧模型或更改 MODEL_NAME 后重新开始。")
            return
    else:
        print("🆕 未找到已有模型，将从头开始训练 (Start From Scratch)")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,  # 建议改回 10，6 稍微有点少
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            # 🟢 修正注释：对应现在的 [256, 256]
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            device=DEVICE,
            tensorboard_log="./ppo_flappy_tensorboard/"
        )

    # 4. 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_path,  # 🟢 修复：现在会保存到 checkpoints 子目录了
        name_prefix="ppo_flappy"
    )

    # 5. 开始训练
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False
        )
        model.save(model_path)
        print("✅ 训练结束")
        print(f"📍 最终模型位置: {os.path.abspath(model_path)}.zip")

    except KeyboardInterrupt:
        print("\n🛑 捕获中断，正在保存模型...")
        model.save(model_path)
        print("✅ 已保存当前进度")


# ================= 👀 测试流程 =================
def test():
    print("👀 加载模型进行测试 (无尽模式)...")

    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ 找不到模型: {model_path}.zip，请先运行 train() 进行训练。")
        return

    # 创建测试环境
    env = gym.make(ENV_ID, render_mode="human", use_lidar=True, background="night")

    try:
        # 🟢 尝试加载，如果之前修改了 Lidar 数量这里会报错
        model = PPO.load(model_path, device=DEVICE)
    except Exception as e:
        print(f"❌ 加载失败！模型输入的形状与当前代码不匹配。")
        print(f"错误信息: {e}")
        print("💡 提示：如果你修改了 Lidar 射线数或网络层数，必须重新训练新模型，不能加载旧的。")
        return

    episode_cnt = 0

    try:
        while True:
            episode_cnt += 1
            obs, _ = env.reset()
            done = False
            score = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                score = info.get('score', 0)
                done = terminated or truncated

            print(f"Episode {episode_cnt} | Score: {score}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n👋 测试结束")
    finally:
        env.close()  # 🟢 修复：现在可以正确关闭环境了


if __name__ == "__main__":
    # 如果你想测试，请确保你有对应当前参数训练出来的模型
    # train()
    test()
