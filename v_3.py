import gymnasium as gym
import flappy_bird_gymnasium
import torch as th  # 🔥 修复：添加 torch 导入
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import os
import glob  # 🔥 修复：用于查找 monitor 文件
import warnings
import matplotlib.pyplot as plt

# 忽略 gymnasium 的特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# ==========================================
# 🎯 核心配置
# ==========================================
MODELS_DIR = "models/flappy_bird_v2"
LOG_DIR = "logs/flappy_bird_v2"
MODEL_NAME = "flappy_bird_master"

# 🔥 关键1：使用向量环境加速训练
N_ENVS = 4  # 并行环境数量
# 🔥 关键2：奖励塑造开关
USE_SHAPED_REWARDS = True

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ==========================================
# 🎁 奖励塑造包装器
# ==========================================
class FlappyBirdRewardShaper(gym.Wrapper):
    """为Flappy Bird添加更密集的奖励信号"""

    def __init__(self, env):
        super().__init__(env)
        self.last_score = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_score = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if USE_SHAPED_REWARDS:
            # 提取关键状态信息 (FlappyBird-v0 simple 模式)
            # obs[0]: bird_y
            # obs[1]: bird_vel
            # obs[2]: pipe_dist_x
            # obs[3]: pipe_top_y
            # obs[4]: pipe_bottom_y

            bird_y = obs[0]
            bird_vel = obs[1]
            pipe_dist_x = obs[2]
            pipe_top_y = obs[3]
            pipe_bottom_y = obs[4]
            gap_center = (pipe_top_y + pipe_bottom_y) / 2

            # 1. 生存奖励
            reward += 0.1

            # 2. 高度保持奖励 (当鸟在管道之间时)
            # pipe_dist_x 范围通常是 [0, width]，需要确认具体数值范围，这里假设标准化处理
            if pipe_dist_x > -0.5 and pipe_dist_x < 0.5:
                height_diff = abs(bird_y - gap_center)
                # 距离中心越近，奖励越高，最大 0.5
                reward += max(0, 0.5 - height_diff)

            # 3. 速度惩罚 (稍微抑制剧烈抖动)
            reward -= abs(bird_vel) * 0.01

        return obs, reward, terminated, truncated, info


# ==========================================
# ⚙️ 动态学习率
# ==========================================
def linear_schedule(initial_value: float):
    """线性下降学习率"""

    def func(progress_remaining: float):
        return progress_remaining * initial_value

    return func


# ==========================================
# 🚀 训练函数
# ==========================================
def train():
    print(f">>> [训练] 初始化 {N_ENVS} 个并行环境...")

    # 定义环境工厂函数
    def make_env():
        env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
        env = FlappyBirdRewardShaper(env)  # 添加奖励塑造
        return env

    # 创建向量化环境 (Monitor 会自动添加到每个子环境)
    env = make_vec_env(make_env, n_envs=N_ENVS, monitor_dir=LOG_DIR)

    final_path = f"{MODELS_DIR}/{MODEL_NAME}.zip"
    ckpt_pattern = f"{MODELS_DIR}/ckpt_*_steps.zip"

    # --- 模型加载逻辑 ---
    if os.path.exists(final_path):
        print(f">>> ♻️ 加载最终模型并继续训练...")
        model = DQN.load(final_path, env=env, tensorboard_log=LOG_DIR)
        start_steps = model.num_timesteps
        current_lr = 1e-5  # 继续训练使用较小学习率

    else:
        # 查找最近的 Checkpoint
        ckpts = glob.glob(ckpt_pattern)
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getctime)
            print(f">>> ♻️ 加载Checkpoint: {os.path.basename(latest_ckpt)}")
            model = DQN.load(latest_ckpt, env=env, tensorboard_log=LOG_DIR)
            start_steps = model.num_timesteps
        else:
            print(">>> 🆕 从零开始训练...")
            start_steps = 0

            # 优化后的网络结构
            policy_kwargs = dict(
                net_arch=[256, 256],
                activation_fn=th.nn.ReLU,  # 🔥 修复：使用 th.nn.ReLU
            )

            model = DQN(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log=LOG_DIR,
                learning_rate=linear_schedule(1e-4),
                buffer_size=500_000,
                learning_starts=10_000,
                batch_size=256,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                exploration_fraction=0.2,  # 前20%时间探索
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                policy_kwargs=policy_kwargs,
            )

    # --- 回调函数 ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // N_ENVS,
        save_path=MODELS_DIR,
        name_prefix="ckpt",
        save_replay_buffer=False,  # 设为False以节省磁盘空间，如果是True会导致ckpt文件巨大
        save_vecnormalize=True,
    )

    # 评估环境 (为了准确评估，建议不加 RewardShaper，看真实分数，但为了保持输入一致性，这里保留结构)
    eval_env = make_vec_env(make_env, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODELS_DIR}/best_model",
        log_path=f"{LOG_DIR}/eval",
        eval_freq=50_000 // N_ENVS,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    TOTAL_STEPS = 2_000_000  # 设置为 2M 步通常足够入门
    steps_to_train = TOTAL_STEPS - start_steps

    if steps_to_train <= 0:
        print(">>> 模型已达到目标步数，无需训练")
        return

    print(f">>> [训练] 目标: {steps_to_train // 1000}k 步 (总 {TOTAL_STEPS // 1000}k)...")

    try:
        model.learn(
            total_timesteps=steps_to_train,
            progress_bar=True,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name="dqn_run",
            reset_num_timesteps=False,
        )
        model.save(f"{MODELS_DIR}/{MODEL_NAME}")
        print(f">>> [训练] 完成！已保存至 {MODEL_NAME}")

    except KeyboardInterrupt:
        print("\n>>> [中断] 保存紧急备份...")
        model.save(f"{MODELS_DIR}/interrupted_model")

    finally:
        env.close()
        eval_env.close()


# ==========================================
# 🧪 测试函数
# ==========================================
def test(episodes=5, deterministic=True):
    best_path = f"{MODELS_DIR}/best_model/best_model.zip"  # EvalCallback 通常保存在子文件夹
    final_path = f"{MODELS_DIR}/{MODEL_NAME}.zip"

    load_path = None
    if os.path.exists(best_path):
        load_path = best_path
        print(f">>> [测试] 加载最佳模型: {best_path}")
    elif os.path.exists(final_path):
        load_path = final_path
        print(f">>> [测试] 加载最终模型: {final_path}")
    else:
        # 查找 ckpt
        ckpts = glob.glob(f"{MODELS_DIR}/ckpt_*.zip")
        if ckpts:
            load_path = max(ckpts, key=os.path.getctime)
            print(f">>> [测试] 加载Checkpoint: {load_path}")

    if not load_path:
        print(">>> ❌ 未找到可加载的模型")
        return

    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    # 注意：测试时不需要 RewardShaper，我们需要看原始分数

    model = DQN.load(load_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # FlappyBird 环境通常在 info 中包含 'score'
            score = info.get('score', score)

        print(f"Episode {ep + 1}: Score = {score}")

    env.close()


# ==========================================
# 📈 绘制训练曲线 (修复版)
# ==========================================
def plot_results():
    print(">>> [绘图] 正在生成曲线...")

    # 🔥 修复：make_vec_env 会生成多个 monitor 文件 (0.monitor.csv, 1.monitor.csv...)
    # 这里的逻辑是读取所有文件并计算平均值，或者只读取第一个
    monitor_files = glob.glob(f"{LOG_DIR}/*.monitor.csv")

    if not monitor_files:
        print(">>> [警告] 未找到 Monitor CSV 文件，跳过绘图。")
        return

    try:
        # 读取第一个 monitor 文件 (通常足够代表趋势)
        # 如果需要更精确，可以聚合所有 dataframe
        df = load_results(LOG_DIR)

        if len(df) < 2:
            print(">>> [警告] 数据点太少，无法绘图。")
            return

        x, y = ts2xy(df, 'timesteps')

        # 平滑处理
        def moving_average(values, window):
            weights = np.repeat(1.0, window) / window
            return np.convolve(values, weights, 'valid')

        if len(y) > 100:
            y_smoothed = moving_average(y, window=50)
            x_smoothed = x[len(x) - len(y_smoothed):]
        else:
            y_smoothed, x_smoothed = y, x

        plt.figure(figsize=(10, 5))
        plt.plot(x_smoothed, y_smoothed, label="Smoothed Reward")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("Flappy Bird Training Progress")
        plt.legend()
        plt.grid(True)

        save_path = f"{LOG_DIR}/training_curve.png"
        plt.savefig(save_path)
        print(f">>> [绘图] 曲线已保存至: {save_path}")
        plt.close()

    except Exception as e:
        print(f">>> [错误] 绘图失败: {e}")


if __name__ == "__main__":
    # 1. 训练
    # train()

    # 2. 测试模型（训练完成后）
    test(episodes=20, deterministic=True)

    # 3. 绘制结果
    plot_results()
