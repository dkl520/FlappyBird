# 导入必要的库
import gymnasium as gym
import flappy_bird_gymnasium
import torch
import os
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# ==========================================
# ⚙️ 全局配置
# ==========================================
MODELS_DIR = "models/flappy_ppo_finalv8"
LOG_DIR = "logs/flappy_ppo_finalv8"
BEST_MODEL_NAME = "best_modelv8"
FINAL_MODEL_NAME = "last_run_modelv8"

N_ENVS = 8  # CPU 核心数允许的话建议用 8，否则 4
TOTAL_TIMESTEPS = 1_500_000

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ==========================================
# 🧠 核心改进：基于原始向量的观测包装器 (已修正索引)
# ==========================================
class SmartObsWrapper(gym.Wrapper):
    """
    修复版：正确解析 FlappyBird-v0 的 12 维观测数组
    索引说明:
    0: 最后一个管道的水平位置
    1: 玩家的垂直位置
    2: 玩家的垂直速度
    3: 下一个管道的水平位置
    4: 下一个上方管道的垂直位置
    5: 下一个下方管道的垂直位置
    6: 下下个管道的水平位置
    7: 下下个上方管道的垂直位置
    8: 下下个下方管道的垂直位置
    9: 玩家的旋转角度
    10: ? (其他信息)
    11: ? (其他信息)
    """

    def __init__(self, env):
        super().__init__(env)
        # 输出：3维向量 [dy, vel, dx]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

    def reset(self, **kwargs):
        # 原始环境返回的 raw_obs 是一个 12 维向量
        raw_obs, info = self.env.reset(**kwargs)
        return self._process_obs(raw_obs), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        new_obs = self._process_obs(raw_obs)
        return new_obs, reward, terminated, truncated, info

    def _process_obs(self, raw_obs):
        """
        从 FlappyBird-v0 的 12 维观测中提取关键信息
        """
        # 1. 提取原始数据 (已修正索引)
        pipe_dist = raw_obs[3]  # 下一个管道的水平位置
        pipe_top = raw_obs[4]   # 下一个上方管道的垂直位置
        pipe_bottom = raw_obs[5] # 下一个下方管道的垂直位置
        bird_y = raw_obs[1]     # 玩家的垂直位置 (修正: 索引1)
        bird_vel = raw_obs[2]   # 玩家的垂直速度 (修正: 索引2)

        # 2. 计算特征
        pipe_center_y = (pipe_top + pipe_bottom) / 2.0

        # A. 垂直偏差 (dy)
        dy = (bird_y - pipe_center_y) / 512.0  # 屏幕高度512

        # B. 速度 (vel)
        vel = bird_vel / 10.0  # 大致最大速度10

        # C. 水平距离 (dx)
        dx = pipe_dist / 288.0  # 屏幕宽度288

        return np.array([dy, vel, dx], dtype=np.float32)


# ==========================================
# 🛡️ 奖励函数：SmartRewardWrapper (逻辑修正)
# ==========================================
class SmartRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # obs 已经是 [dy, vel, dx]
        dy = obs[0]
        dx = obs[2]

        # 1. 基础存活奖励
        reward += 0.01

        # 2. 智能居中奖励 (只在管子在小鸟前面时应用)
        if dx > 0:  # 仅当管子在小鸟前面
            center_bonus = 0.2 * (1.0 - abs(dy) * 4)
            reward += max(0, center_bonus)

        # 3. 撞击惩罚
        if terminated:
            reward -= 1.0

        return obs, reward, terminated, truncated, info


# ==========================================
# 🛠️ 环境组装
# ==========================================
def make_env():
    # 基础环境 (已移除无效的 use_lidar 参数)
    env = gym.make("FlappyBird-v0", render_mode=None)
    # 1. 包装观测
    env = SmartObsWrapper(env)
    # 2. 包装奖励
    env = SmartRewardWrapper(env)
    return env


# ==========================================
# 🏋️‍♂️ 训练流程
# ==========================================
def train():
    print(f"\n>>> [最终修复版] 启动训练...")
    print(f">>> 模式: CPU (加速) | 观测: [dy, vel, dx]")

    # 重新创建环境，避免旧缓存问题
    env = make_vec_env(make_env, n_envs=N_ENVS, monitor_dir=LOG_DIR)
    eval_env = make_vec_env(make_env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",  # 强制 CPU
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.Tanh,
        ),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=LOG_DIR,
        eval_freq=20000 // N_ENVS,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("收到中断信号，正在保存...")

    model.save(os.path.join(MODELS_DIR, FINAL_MODEL_NAME))
    env.close()
    eval_env.close()
    print("训练结束！")


# ==========================================
# 🎮 测试流程
# ==========================================
def test():
    model_path = os.path.join(MODELS_DIR, f"{BEST_MODEL_NAME}.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, f"{FINAL_MODEL_NAME}.zip")

    if not os.path.exists(model_path):
        print("❌ 没找到模型文件，请先运行 train()")
        return

    print(f">>> 🎮 加载模型: {model_path}")

    env = gym.make("FlappyBird-v0", render_mode="human")
    env = SmartObsWrapper(env)

    model = PPO.load(model_path, device="cpu")

    for ep in range(10):
        obs, _ = env.reset()
        done = False
        print(f"--- 第 {ep + 1} 局 ---")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if terminated:
                print(f"💀 死亡! 分数: {info.get('score', 0)}")
                time.sleep(1.0)

    env.close()


if __name__ == "__main__":
    train()
    # test()  # 测试时取消注释