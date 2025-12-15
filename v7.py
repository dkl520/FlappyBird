import gymnasium as gym
import flappy_bird_gymnasium
import torch
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# ==========================================
# ⚙️ 全局配置
# ==========================================
MODELS_DIR = "models/flappy_ppo_hard"  # 改个名字，区分之前的版本
LOG_DIR = "logs/flappy_ppo_hard"
BEST_MODEL_NAME = "best_model"
FINAL_MODEL_NAME = "last_run_model"

N_ENVS = 4
TOTAL_TIMESTEPS = 1_000_000

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ==========================================
# 🔥 核心修改：自定义“严格模式”包装器
# ==========================================
class StrictSafetyWrapper(gym.Wrapper):
    """
    这是一个严格的教练：
    1. 如果由于离管子太近而过关，没收大部分奖励。
    2. 每一帧如果离障碍物太近，给予微小惩罚。
    """

    def __init__(self, env, safe_dist=0.14):
        super().__init__(env)
        self.safe_dist = safe_dist  # 安全距离阈值 (0.0 - 1.0)

    def step(self, action):
        # 获取原始环境的反馈
        # obs 在 use_lidar=True 时，是一个包含 180 个雷达数据的数组
        # 数值越小，代表离障碍物越近
        obs, reward, terminated, truncated, info = self.env.step(action)

        # === 😈 魔改奖励逻辑 ===

        # 1. 获取当前最近障碍物的距离
        # obs 里的数据通常是归一化的距离
        min_distance = np.min(obs)

        # 2. 惩罚“贴脸飞行” (Proximity Penalty)
        # 如果离任何东西太近（小于阈值），每一帧都扣一点点分
        # 这会逼迫鸟时刻保持在空旷地带
        if min_distance < self.safe_dist:
            reward -= 0.05  # 微小惩罚，不要扣太多，否则它会选择自杀

        # 3. 惩罚“惊险过关”
        # 默认过管奖励通常是 1.0 (具体看库版本，通常通过 info['score'] 变化判断也可以)
        # 这里我们简单假设 reward > 0.5 就是过管了
        if reward >= 1.0:
            if min_distance < self.safe_dist:
                # 刚才虽然过管了，但是离管子太近了！
                # 把奖励打折，只给 0.2 分
                reward = 0.2
                # 或者：reward -= 0.8

        return obs, reward, terminated, truncated, info


# ==========================================
# 🛠️ 环境构建函数
# ==========================================
def make_env():
    # 1. 基础环境
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=True)

    # 2. 🔥 套上我们的严格教练包装器
    env = StrictSafetyWrapper(env, safe_dist=0.15)  # 0.2 表示雷达探测距离的 20%

    return env


# ==========================================
# 🏋️‍♂️ 训练主函数 (保持不变，逻辑通用)
# ==========================================
def train():
    print(f"\n>>> [严格模式] 启动训练，如果飞得太贴近管子会被扣分！")
    print(f">>> 目标步数: {TOTAL_TIMESTEPS}")

    env = make_vec_env(make_env, n_envs=N_ENVS, monitor_dir=LOG_DIR)
    eval_env = make_vec_env(make_env, n_envs=1)

    # 载入历史最佳分数的逻辑
    best_model_path = os.path.join(MODELS_DIR, f"{BEST_MODEL_NAME}.zip")
    historical_best_score = -np.inf

    if os.path.exists(best_model_path):
        print(f">>> 🏆 发现历史模型，正在评估...")
        try:
            temp_model = PPO.load(best_model_path)
            mean_reward, _ = evaluate_policy(temp_model, eval_env, n_eval_episodes=5)
            historical_best_score = mean_reward
            print(f">>> 📊 历史最高分: {historical_best_score:.2f}")
            del temp_model
        except:
            pass

    # 模型定义
    final_model_path = os.path.join(MODELS_DIR, f"{FINAL_MODEL_NAME}.zip")
    if os.path.exists(final_model_path):
        print(">>> ♻️ 继续训练...")
        model = PPO.load(final_model_path, env=env)
    else:
        print(">>> ✨ 新建模型...")
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
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=torch.nn.Tanh),
        )

    # 回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=LOG_DIR,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    eval_callback.best_mean_reward = historical_best_score

    # 开始训练
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True,
                    reset_num_timesteps=False)
    except KeyboardInterrupt:
        print(">>> 中断保存...")

    model.save(final_model_path)
    env.close()
    eval_env.close()


# ==========================================
# 🎮 测试函数
# ==========================================
import  time
def test():
    # ... (加载模型路径部分保持不变) ...
    load_path = os.path.join(MODELS_DIR, f"{BEST_MODEL_NAME}.zip")
    if not os.path.exists(load_path):
        load_path = os.path.join(MODELS_DIR, f"{FINAL_MODEL_NAME}.zip")

    if not os.path.exists(load_path):
        print("❌ 无模型")
        return

    print(f">>> 🎮 正在加载模型: {load_path}")
    print(f">>> 🕵️ 死亡暂停模式已开启：撞死后请看画面，按回车继续！")

    # 1. 创建环境
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    model = PPO.load(load_path)

    for ep in range(10):  # 测试 10 局
        obs, _ = env.reset()
        done = False
        start_time = time.time()

        print(f"\n🎬 第 {ep + 1} 局开始...")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # 无限模式逻辑
            done = terminated

            # 如果触发 Gym 的时间限制，这里忽略它，只看撞死
            if truncated:
                pass

            # 🔥🔥🔥 核心修改：死亡暂停 🔥🔥🔥
            if terminated:
                # 计算这局飞了多久
                duration = time.time() - start_time
                final_score = info.get('score', 0)

                print(f"🛑 [撞车瞬间]！")
                print(f"   分数: {final_score} | 存活: {duration:.2f}秒")
                print(f"   👀 请检查游戏窗口，看看到底撞到了哪里（顶部？底部？管子边缘？）")

                # ⏸️ 这里会让程序卡住，直到你按下回车
                input("👉 按 [回车键 Enter] 开始下一局...")

    env.close()


if __name__ == "__main__":
    # train()
    test()