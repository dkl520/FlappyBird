import gymnasium as gym
import flappy_bird_gymnasium
import torch
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# ==========================================
# ⚙️ 全局配置
# ==========================================
# 文件夹路径
MODELS_DIR = "models/flappy_ppo_final"
LOG_DIR = "logs/flappy_ppo_final"

# 文件名区分
# BEST_MODEL_NAME: 永远只存历史最高分（巅峰状态）
# FINAL_MODEL_NAME: 存最后一次训练结束时的状态（哪怕变笨了也存这里，用于续训）
BEST_MODEL_NAME = "best_model"
FINAL_MODEL_NAME = "last_run_model"

# 并行环境数 (根据CPU核心数调整)
N_ENVS = 4
# 总训练步数
TOTAL_TIMESTEPS = 1_000_000

# 创建目录
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ==========================================
# 🛠️ 环境构建函数
# ==========================================
def make_env():
    # use_lidar=True 是关键，开启雷达探测
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=True)
    return env


# ==========================================
# 🏋️‍♂️ 训练主函数
# ==========================================
def train():
    print(f"\n>>> [系统启动] 准备开始训练，目标步数: {TOTAL_TIMESTEPS}")

    # 1. 创建并行的训练环境
    env = make_vec_env(make_env, n_envs=N_ENVS, monitor_dir=LOG_DIR)

    # 2. 准备评估环境 (用于测试当前模型好坏，决定是否保存 best_model)
    eval_env = make_vec_env(make_env, n_envs=1)

    # ======================================================
    # 🔥 核心逻辑：防止重启训练时覆盖掉历史最高分
    # ======================================================
    best_model_path = os.path.join(MODELS_DIR, f"{BEST_MODEL_NAME}.zip")
    historical_best_score = -np.inf  # 默认负无穷

    if os.path.exists(best_model_path):
        print(f">>> 🏆 发现历史 'best_model.zip'，正在测试它的含金量...")
        try:
            # 加载旧的巅峰模型，跑 5 局看看它到底多少分
            temp_model = PPO.load(best_model_path)
            mean_reward, _ = evaluate_policy(temp_model, eval_env, n_eval_episodes=5)
            historical_best_score = mean_reward
            print(f">>> 📊 确认历史最高纪录: {historical_best_score:.2f} 分")
            print("    (只有新模型超过这个分数，才会覆盖 best_model.zip)")
            del temp_model  # 释放内存
        except Exception as e:
            print(f">>> ⚠️ 历史模型读取失败，将重新开始记录。错误: {e}")
    else:
        print(">>> 🆕 没有发现历史记录，将建立新的排行榜。")

    # ======================================================
    # 🧠 模型初始化
    # ======================================================
    final_model_path = os.path.join(MODELS_DIR, f"{FINAL_MODEL_NAME}.zip")

    if os.path.exists(final_model_path):
        print(f">>> ♻️ 发现上次中断的进度 '{FINAL_MODEL_NAME}'，正在加载继续训练...")
        # reset_num_timesteps=False 表示接上传次的时间步继续计数
        model = PPO.load(final_model_path, env=env)
    else:
        print(f">>> ✨ 创建全新的 PPO 模型...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # 熵系数，增加探索
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=torch.nn.Tanh
            ),
        )

    # ======================================================
    # 📝 回调设置 (自动保存最高分)
    # ======================================================
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,  # 发现新高分时，自动存到这里
        log_path=LOG_DIR,
        eval_freq=10000,  # 每 1万步 评估一次
        n_eval_episodes=5,  # 每次评估跑 5 局取平均
        deterministic=True,  # 评估时使用确定性策略（不乱试）
        render=False
    )

    # 🔥 关键修改：告诉回调函数目前的最高纪录是多少
    # 这样新的一轮训练如果只有 50 分，就不会覆盖掉之前 300 分的模型
    eval_callback.best_mean_reward = historical_best_score

    # ======================================================
    # 🚀 开始学习
    # ======================================================
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=eval_callback,
            progress_bar=True,
            reset_num_timesteps=False  # 如果是续训，不重置总步数
        )
        print(">>> ✅ 训练目标达成！")
    except KeyboardInterrupt:
        print("\n>>> 🛑 用户手动中断训练！正在保存当前进度...")

    # ======================================================
    # 💾 保存“最后一次”的状态 (无论好坏)
    # ======================================================
    # 这个文件用于下次 'Resume' 继续训练
    model.save(final_model_path)
    print(f">>> 💾 进度已保存至: {final_model_path}")
    print(f">>> 🌟 历史最强模型 (请测试这个): {best_model_path}")

    env.close()
    eval_env.close()


# ==========================================
# 🎮 测试/展示函数
# ==========================================
def test():
    # 永远优先加载 best_model，因为那才是我们的巅峰
    load_path = os.path.join(MODELS_DIR, f"{BEST_MODEL_NAME}.zip")

    if not os.path.exists(load_path):
        print(f"⚠️ 没找到巅峰模型，尝试加载最后一次的模型: {FINAL_MODEL_NAME}.zip")
        load_path = os.path.join(MODELS_DIR, f"{FINAL_MODEL_NAME}.zip")

    if not os.path.exists(load_path):
        print("❌ 没有任何模型文件！请先运行 train()。")
        return

    print(f"\n>>> 🎮 正在加载模型进行演示: {load_path}")

    # 开启 render_mode='human' 让你看到画面
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)

    model = PPO.load(load_path)

    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            # 必须 deterministic=True，否则 AI 会在这个阶段尝试随机动作导致撞死
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score = info.get('score', score)  # FlappyBird 环境会在 info 里返回 score

        print(f"第 {ep + 1} 局得分: {score}")

    env.close()


if __name__ == "__main__":
    # ==========================================
    # 👇 控制开关 👇
    # ==========================================

    # 1. 想要训练时，解开下面这行的注释：
    train()

    # 2. 想要看 AI 玩游戏时，注释掉上面的 train()，解开下面这行：
    test()