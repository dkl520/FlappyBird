# 导入必要的库
import gymnasium as gym
import flappy_bird_gymnasium  # Flappy Bird 的 Gymnasium 环境
import torch
import os
import numpy as np
from stable_baselines3 import PPO  # 使用 PPO 算法
from stable_baselines3.common.env_util import make_vec_env  # 用于并行环境
from stable_baselines3.common.callbacks import EvalCallback  # 评估回调
from stable_baselines3.common.evaluation import evaluate_policy  # 用于评估策略

# ==========================================
# ⚙️ 全局配置：定义训练和日志路径、超参数等
# ==========================================
MODELS_DIR = "models/flappy_ppo_hard"  # 模型保存目录（区别于普通版本）
LOG_DIR = "logs/flappy_ppo_hard"       # TensorBoard 日志目录
BEST_MODEL_NAME = "best_model"         # 最佳模型文件名（自动保存）
FINAL_MODEL_NAME = "last_run_model"    # 最终模型文件名（每次训练结束保存）

N_ENVS = 4                             # 并行环境数量（提升采样效率）
TOTAL_TIMESTEPS = 1_000_000            # 总训练步数

# 创建模型和日志目录（如果不存在）
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

    def __init__(self, env, safe_dist=0.25):
        super().__init__(env)
        self.safe_dist = safe_dist  # 安全距离阈值（归一化值，0.0~1.0）

    def step(self, action):
        # 获取原始环境返回的观测、奖励、终止状态等
        # obs 在 use_lidar=True 时，是一个包含 180 个雷达数据的数组
        # 数值越小，代表离障碍物越近（0.0 表示碰撞，1.0 表示很远）
        obs, reward, terminated, truncated, info = self.env.step(action)

        # === 😈 魔改奖励逻辑 ===

        # 1. 获取当前最近障碍物的距离（所有雷达方向中的最小值）
        min_distance = np.min(obs)  # 可能来自上方（上管）、下方（下管）或前方

        # 2. 惩罚“贴脸飞行” (Proximity Penalty)
        # 如果离任何障碍物太近（小于安全阈值），每帧扣一点分
        # 目的是鼓励智能体保持在空旷区域，避免擦边飞行
        if min_distance < self.safe_dist:
            reward -= 0.03  # 微小惩罚，避免智能体因惩罚过重而“主动自杀”

        # 3. 惩罚“惊险过关”
        # 默认环境中，成功穿过一个管道会获得 reward >= 1.0
        # 如果此时离障碍物仍太近（min_distance < safe_dist），说明是“危险通关”
        if reward >= 1.0:
            if min_distance < self.safe_dist:
                # 虽然过了管子，但太危险！只给少量奖励（0.2 分）
                reward -= 0.4
                # 注：也可用 reward -= 0.8，效果类似

        # 返回修改后的结果
        return obs, reward, terminated, truncated, info


# ==========================================
# 🛠️ 环境构建函数：用于创建单个训练环境
# ==========================================
def make_env():
    # 1. 创建基础 Flappy Bird 环境，使用 LiDAR 观测（180 维向量）
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=True)

    # 2. 🔥 套上我们的严格教练包装器，设置安全距离为 0.2（即 20% 的探测范围）
    env = StrictSafetyWrapper(env, safe_dist=0.2)

    return env


# ==========================================
# 🏋️‍♂️ 训练主函数：支持断点续训 + 自动保存最佳模型
# ==========================================
def train():
    print(f"\n>>> [严格模式] 启动训练，如果飞得太贴近管子会被扣分！")
    print(f">>> 目标步数: {TOTAL_TIMESTEPS}")

    # 创建 N_ENVS 个并行环境，用于高效采样
    env = make_vec_env(make_env, n_envs=N_ENVS, monitor_dir=LOG_DIR)
    # 创建单独的评估环境（不参与训练）
    eval_env = make_vec_env(make_env, n_envs=1)

    # 尝试加载历史最佳模型，用于初始化 best_mean_reward
    best_model_path = os.path.join(MODELS_DIR, f"{BEST_MODEL_NAME}.zip")
    historical_best_score = -np.inf  # 初始化为负无穷

    if os.path.exists(best_model_path):
        print(f">>> 🏆 发现历史模型，正在评估...")
        try:
            temp_model = PPO.load(best_model_path)
            # 用 5 局评估平均得分作为历史最佳
            mean_reward, _ = evaluate_policy(temp_model, eval_env, n_eval_episodes=5)
            historical_best_score = mean_reward
            print(f">>> 📊 历史最高分: {historical_best_score:.2f}")
            del temp_model  # 释放内存
        except Exception as e:
            print(f">>> ⚠️ 加载历史模型失败: {e}")
            pass

    # 决定是新建模型还是继续训练
    final_model_path = os.path.join(MODELS_DIR, f"{FINAL_MODEL_NAME}.zip")
    if os.path.exists(final_model_path):
        print(">>> ♻️ 继续训练...")
        model = PPO.load(final_model_path, env=env)  # 加载并绑定新环境
    else:
        print(">>> ✨ 新建模型...")
        # 定义 PPO 模型结构和超参数
        model = PPO(
            "MlpPolicy",               # 使用全连接网络（MLP）
            env,
            verbose=1,                 # 打印训练日志
            tensorboard_log=LOG_DIR,   # TensorBoard 日志路径
            learning_rate=3e-4,        # 学习率
            n_steps=2048,              # 每次更新收集的步数
            batch_size=64,             # 训练批次大小
            n_epochs=10,               # 每批数据训练轮数
            gamma=0.99,                # 折扣因子
            gae_lambda=0.95,           # GAE 参数
            clip_range=0.2,            # PPO 的裁剪范围
            ent_coef=0.01,             # 熵正则化系数（鼓励探索）
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),  # 策略和价值网络结构
                activation_fn=torch.nn.Tanh  # 激活函数
            ),
        )

    # 设置评估回调：每 10000 步评估一次，自动保存最佳模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=LOG_DIR,
        eval_freq=10000 // N_ENVS,     # 注意：eval_freq 是按每个环境的步数计算
        n_eval_episodes=5,             # 每次评估跑 5 局
        deterministic=True,            # 使用确定性动作（关闭探索）
        render=False                   # 不渲染评估过程
    )
    # 手动设置历史最佳分数，避免覆盖已有最佳模型
    eval_callback.best_mean_reward = historical_best_score

    # 开始训练
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=eval_callback,
            progress_bar=True,          # 显示进度条
            reset_num_timesteps=False   # 续训时不重置 timestep 计数
        )
    except KeyboardInterrupt:
        print(">>> 中断保存...")

    # 保存最终模型
    model.save(final_model_path)
    env.close()
    eval_env.close()


# ==========================================
# 🎮 测试函数：可视化运行 + 死亡暂停功能（便于分析失败原因）
# ==========================================
import time  # 用于计算每局存活时间


def test():
    # 尝试加载最佳模型，若无则加载最终模型
    load_path = os.path.join(MODELS_DIR, f"{BEST_MODEL_NAME}.zip")
    if not os.path.exists(load_path):
        load_path = os.path.join(MODELS_DIR, f"{FINAL_MODEL_NAME}.zip")

    if not os.path.exists(load_path):
        print("❌ 无模型")
        return

    print(f">>> 🎮 正在加载模型: {load_path}")
    print(f">>> 🕵️ 死亡暂停模式已开启：撞死后请看画面，按回车继续！")

    # 创建可渲染的人类可视化环境（render_mode="human"）
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    model = PPO.load(load_path)

    # 运行 10 局测试
    for ep in range(10):
        obs, _ = env.reset()  # 重置环境
        done = False
        start_time = time.time()  # 记录开始时间

        print(f"\n🎬 第 {ep + 1} 局开始...")

        while not done:
            # 使用模型预测动作（确定性策略）
            action, _ = model.predict(obs, deterministic=True)
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)

            # 只有撞到障碍物才算真正结束（忽略时间截断）
            done = terminated

            # 如果环境因最大步数截断（truncated），我们忽略它，继续飞
            if truncated:
                pass

            # 🔥🔥🔥 核心功能：死亡后暂停，等待用户确认
            if terminated:
                duration = time.time() - start_time  # 计算存活时间
                final_score = info.get('score', 0)   # 获取最终得分（通过的管道数）

                print(f"🛑 [撞车瞬间]！")
                print(f"   分数: {final_score} | 存活: {duration:.2f}秒")
                print(f"   👀 请检查游戏窗口，看看到底撞到了哪里（顶部？底部？管子边缘？）")

                # ⏸️ 程序暂停，等待用户按回车键继续下一局
                input("👉 按 [回车键 Enter] 开始下一局...")

    # 关闭环境
    env.close()


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # train()  # 如需训练，取消注释此行
    test()     # 当前默认运行测试