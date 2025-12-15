import os
import time
import gymnasium as gym
import flappy_bird_gymnasium
import torch
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# ==========================================
# ⚙️ 全局配置
# ==========================================
MODELS_DIR = "models/flappy_ppo_v3"
LOG_DIR = "logs/flappy_ppo_v3"
MODEL_NAME = "flappy_bird_ppo_best"
N_ENVS = 4

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env():
    """创建单个 FlappyBird 环境（LIDAR 模式）"""
    return gym.make("FlappyBird-v0", render_mode=None, use_lidar=True)


def train():
    print(f">>> [初始化] 启动 {N_ENVS} 个并行环境 (PPO)...")
    env = make_vec_env(make_env, n_envs=N_ENVS, monitor_dir=LOG_DIR)

    # 🔧 推荐超参数（针对 Flappy Bird 稀疏奖励优化）
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,

        # 学习率稍低，避免在稀疏奖励下震荡
        learning_rate=2.5e-4,
        n_steps=2048,  # 总 buffer = 2048 * 4 = 8192
        batch_size=128,  # 增大批次提升稳定性
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # 保持适度探索

        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # 稍大网络，应对184维输入
            activation_fn=torch.nn.Tanh
        ),
        seed=42  # 可复现性
    )

    # 🔄 断点续训
    final_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.zip")
    if os.path.exists(final_path):
        print(">>> ♻️ 加载已有模型继续训练...")
        model = PPO.load(final_path, env=env, tensorboard_log=LOG_DIR)

    # 📊 评估回调（每 5k 步评估一次，更快反馈）
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=LOG_DIR,
        eval_freq=5000,  # 更频繁评估
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    TOTAL_STEPS = 1_000_000
    print(f">>> [开始训练] 目标步数: {TOTAL_STEPS}")

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=eval_callback,
            progress_bar=True
        )
        model.save(final_path)
        print(">>> ✅ 训练完成！")

    except KeyboardInterrupt:
        print(">>> 🛑 用户中断，保存当前模型...")
        model.save(os.path.join(MODELS_DIR, "interrupted_ppo"))
    finally:
        env.close()


def test(render=True, episodes=10):
    """测试模型，支持渲染和死亡截图"""
    best_path = os.path.join(MODELS_DIR, "best_model.zip")
    final_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.zip")

    model_path = best_path if os.path.exists(best_path) else final_path
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ 未找到模型文件！请先训练。")

    print(f">>> 🎮 加载模型: {model_path}")

    # 根据是否渲染选择 render_mode
    render_mode = "rgb_array" if render else None
    env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=True)
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        last_frame = None

        while not done:
            if render:
                frame = env.render()
                last_frame = frame
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Flappy Bird Replay", bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score = info.get('score', score)

        if render and last_frame is not None:
            timestamp = int(time.time())
            filename = os.path.join(LOG_DIR, f"death_ep{ep + 1}_score{score}_{timestamp}.png")
            cv2.imwrite(filename, cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
            print(f"💀 第 {ep + 1} 局结束 (分: {score}) → 截图: {filename}")

    env.close()
    if render:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="启动训练")
    parser.add_argument("--test", action="store_true", help="启动测试")
    parser.add_argument("--no-render", action="store_true", help="测试时不渲染画面")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.test:
        test(render=not args.no_render)
    else:
        print("用法: python flappy_ppo.py --train   # 开始训练")
        print("      python flappy_ppo.py --test    # 观看 AI 玩游戏")
        print("      python flappy_ppo.py --test --no-render  # 无渲染测试（仅打印分数）")