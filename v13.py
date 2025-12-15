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
MODEL_NAME = "ppo_flappy_final13"  # 不带后缀
TOTAL_TIMESTEPS = 1_000_000
DEVICE = "cpu"  # 保持你原来的设置


# ================= 🛡️ 安全奖励包装器 (保留原逻辑) =================
class StrictSafetyWrapper(gym.Wrapper):
    def __init__(self, env, safe_dist=0.20):
        super().__init__(env)
        self.safe_dist = safe_dist

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # SB3 的环境通常会自动处理 obs，但在单环境 Wrapper 中 obs 还是 numpy 数组
        # 你的逻辑：惩罚贴管飞行
        if np.min(obs) < self.safe_dist:
            reward += 0.05

        return obs, reward, terminated, truncated, info


# ================= 🏃 训练流程 =================
def train():
    print("🚀 准备开始训练 (Stable-Baselines3 版)...")

    # 1. 创建环境 (Monitor 用于记录数据给 SB3)
    env = gym.make(ENV_ID, use_lidar=True, background=None)
    env = StrictSafetyWrapper(env, safe_dist=0.20)
    env = Monitor(env)

    # 2. 路径处理
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    checkpoint_path = os.path.join(MODEL_DIR, "checkpoints")

    # 3. 断点续训逻辑
    # SB3 保存的模型后缀是 .zip，所以我们需要检查 .zip 文件
    if os.path.exists(f"{model_path}.zip"):
        print(f"🔄 发现上次训练模型: {model_path}.zip")
        # 加载旧模型，并绑定当前环境
        model = PPO.load(model_path, env=env, device=DEVICE)
        print("✅ 模型加载成功！将在该模型基础上继续训练 (Resume Training)")
    else:
        print("🆕 未找到已有模型，将从头开始训练 (Start From Scratch)")
        # 初始化新模型 (参数映射你的原配置)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=2.5e-4,
            n_steps=2048,  # UPDATE_TIMESTEP
            batch_size=64,
            n_epochs=10,  # K_EPOCHS
            gamma=0.99,
            gae_lambda=0.95,  # LAMBDA
            clip_range=0.2,  # EPS_CLIP
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[128, 128]),  # 对应你的两个 128 层
            verbose=1,
            device=DEVICE,
            tensorboard_log="./ppo_flappy_tensorboard/"
        )

    # 4. 回调函数：定期保存 (替代你原来的 if time_step % 50000 == 0)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=MODEL_DIR,
        name_prefix="ppo_flappy"
    )

    # 5. 开始训练 (progress_bar=True 自带你的进度条需求)
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False  # 续训时不重置步数计数器
        )
        # 最终保存
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

    # 路径
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ 找不到模型: {model_path}.zip，请先运行 train() 进行训练。")
        return

    # 创建测试环境 (Render Mode)
    env = gym.make(ENV_ID, render_mode="human", use_lidar=True, background="night")

    # 加载模型
    model = PPO.load(model_path, device=DEVICE)

    episode_cnt = 0

    while True:
        episode_cnt += 1
        obs, _ = env.reset()
        done = False
        score = 0
        step_cnt = 0

        while not done:
            # predict 返回 (action, state)，这里只需要 action
            # deterministic=True 让测试表现更稳定
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            score = info.get('score', 0)
            step_cnt += 1

            done = terminated or truncated

        print(f"Episode {episode_cnt} | Score: {score} | Steps: {step_cnt}")

        # 暂停 1 秒让你看清 (保留你的习惯)
        time.sleep(1)

    env.close()


if __name__ == "__main__":
    # train() # 训练模式
    test()  # 测试模式