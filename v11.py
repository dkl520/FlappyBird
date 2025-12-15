import os
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
import torch

# ================= 🚀 配置区域 =================
ENV_ID = "FlappyBird-v0"
MODEL_DIR = "trained_models"
LOG_DIR = "logs"
MODEL_NAME = "FlappyBird_Master"
TOTAL_TIMESTEPS = 2_000_000  # 建议由200万步起练
STACK_NUM = 4  # 核心：堆叠 4 帧

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env(render_mode=None):
    def _init():
        env = gym.make(
            ENV_ID,
            render_mode=render_mode,
            use_lidar=True,
            background=None
        )
        return env

    return _init


def train():
    print("🚀 开始训练 (Frame Stacking 版)...")
    print("💡 提示：如果没看到绿色进度条，请确保运行了: pip install tqdm rich")

    num_cpu = 4
    # 1. 创建基础环境
    env = SubprocVecEnv([make_env() for _ in range(num_cpu)])

    # 2. 把环境包在 VecFrameStack 里
    env = VecFrameStack(env, n_stack=STACK_NUM)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log=LOG_DIR,
        device="cpu"  # <---【修改点1】强制使用 CPU，消除警告且速度更快
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // num_cpu,
        save_path=MODEL_DIR,
        name_prefix="ppo_stacked"
    )

    # <---【修改点2】progress_bar=True 已启用
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, progress_bar=True)

    final_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(final_path)
    print(f"✅ 训练完成！模型已保存至: {final_path}")
    env.close()


def test():
    print("👀 正在加载 Frame Stacking 模型...")
    model_path = os.path.join(MODEL_DIR, MODEL_NAME + ".zip")

    if not os.path.exists(model_path):
        print(f"❌ 找不到 {model_path}")
        return

    # 测试环境也要做同样的包裹
    env = DummyVecEnv([make_env(render_mode="human")])
    env = VecFrameStack(env, n_stack=STACK_NUM)  # 测试时必须堆叠

    # 加载模型
    model = PPO.load(model_path, env=env, device="cpu")  # 测试时也建议指定 CPU

    episodes = 5
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()

            total_reward += reward[0]

            if done[0]:
                print(f"🏁 Episode {ep + 1} 结束，得分: {info[0].get('score', 'unknown')}")
                break

    env.close()


if __name__ == "__main__":
    # train()  # 先跑 Train
    test()  # 再跑 Test