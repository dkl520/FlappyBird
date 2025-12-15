import gymnasium as gym
import flappy_bird_gymnasium
import torch
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# ==========================================
# ⚙️ 全局配置参数（每个参数都注释）
# ==========================================
# 模型和日志的保存路径，用于持久化训练成果
MODELS_DIR = "models/flappy_ppo_v3"  # 字符串：模型保存目录
LOG_DIR = "logs/flappy_ppo_v3"  # 字符串：日志（例如 TensorBoard）的保存目录
MODEL_NAME = "flappy_bird_ppo_best"  # 字符串：模型文件名（不含扩展名）

# 并行环境数量。
# int：要创建的并行环境数量（用于向量化环境）。增多可加速样本收集但会占用更多CPU/内存。
N_ENVS = 4

# 如果目录不存在则创建。exist_ok=True 表示如果目录已存在，不会抛出异常。
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ==========================================
# 🚀 环境工厂函数（每个参数注释）
# ==========================================
def make_env():
    """
    创建游戏环境实例。
    返回：gym.Env 对象（未向量化）
    """
    # env_id: 字符串，代表环境注册时的 ID（在 flappy_bird_gymnasium 中为 "FlappyBird-v0"）
    # render_mode=None：不渲染画面（训练时常用 headless 模式以提高速度）。可选值通常包括 'human'、'rgb_array' 等。
    # use_lidar=True：布尔值，指示环境是否返回 lidar（激光雷达）观测；
    #     - True: 额外返回 180 根激光束距离数据（更接近视觉/感知输入），
    #     - False: 仅返回纯数值坐标信息（位置/速度等）。
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=True)
    return env


def train():
    print(f">>> [初始化] 启动 {N_ENVS} 个并行环境 (PPO算法)...")

    # make_vec_env: SB3 的辅助函数，用来把单个环境函数包装成向量化环境。
    # 参数说明：
    #   - make_env: 可调用对象（工厂函数），每次子进程/环境会调用它来创建单个环境实例
    #   - n_envs=N_ENVS: int，要创建的并行环境数量
    #   - monitor_dir=LOG_DIR: 可选，Monitor 将会把 episode 信息（如 reward、length）写到这个目录下
    env = make_vec_env(make_env, n_envs=N_ENVS, monitor_dir=LOG_DIR)

    # ==========================================
    # 🧠 PPO 算法核心配置（每个超参数都注释）
    # ==========================================
    model = PPO(
        "MlpPolicy",  # policy：字符串或策略类，"MlpPolicy" 使用多层感知机（全连接网络）处理观测
        env,  # env：训练环境（向量化环境）
        verbose=1,  # verbose：日志级别（0 无日志，1 基本，2 更详细）
        tensorboard_log=LOG_DIR,  # tensorboard_log：字符串，TensorBoard 日志保存目录

        # --- 超参数调优 ---
        learning_rate=3e-4,  # float：学习率（步长大小）。常见范围 1e-5 ~ 1e-3
        n_steps=2048,  # int：每个环境收集多少步后进行一次优化（总 buffer = n_steps * n_envs）
        batch_size=64,  # int：优化时从 buffer 中采样的最小批次大小
        n_epochs=10,  # int：在同一批数据上重复优化的次数（越大越充分但越慢）
        gamma=0.99,  # float：折扣因子，决定未来奖励的重要性（接近1更重视长期回报）
        gae_lambda=0.95,  # float：GAE 的 lambda，平衡 bias-variance
        clip_range=0.2,  # float 或 callable：PPO 的截断范围，限制新旧策略比率的变化

        ent_coef=0.01,  # float：熵系数，用于鼓励策略探索（熵越大，探索越多）

        # --- 网络结构 ---
        policy_kwargs=dict(
            # net_arch：策略网络和价值网络的结构
            #   - pi=[128,128]：策略网络（Actor）两层隐藏层，分别 128 个神经元
            #   - vf=[128,128]：价值网络（Critic）两层隐藏层，分别 128 个神经元
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=torch.nn.Tanh  # 激活函数：Tanh。可替换为 torch.nn.ReLU 等
        ),
    )

    # --- 断点续训逻辑 ---
    final_path = f"{MODELS_DIR}/{MODEL_NAME}.zip"  # 字符串：最后保存模型的完整路径（含 .zip）
    if os.path.exists(final_path):
        print(f">>> ♻️ 检测到旧模型，正在加载继续训练...")
        # load(path, env=env)：从文件加载模型权重并绑定到新的 env（如果提供）
        model = PPO.load(final_path, env=env)

    # --- 评估回调函数 ---
    # 需要一个单独的、干净的环境来做定期评估（不并行），以更稳定地衡量策略表现
    eval_env = make_vec_env(make_env, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,  # 最佳模型自动保存路径（保存为 best_model.zip）
        log_path=LOG_DIR,  # 评估日志保存路径
        eval_freq=10000,  # int：每训练多少个环境步（timesteps）进行一次评估
        n_eval_episodes=5,  # int：每次评估跑多少个 episode 并取平均
        deterministic=True,  # bool：评估时是否使用确定性策略（关闭探索）
        render=False  # bool 或渲染模式：评估时是否渲染（这里设 False）
    )

    TOTAL_STEPS = 1_000_000  # int：训练的总时间步数（environment steps）
    print(f">>> [开始] 目标训练步数: {TOTAL_STEPS}...")

    try:
        # model.learn：开始训练。
        # 参数：
        #   - total_timesteps=TOTAL_STEPS: int，总训练步数
        #   - callback=eval_callback: 回调对象或列表，会在训练过程中被定期调用
        #   - progress_bar=True: bool，是否显示进度条（仅在某些版本的 SB3 中有效）
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=eval_callback,
            progress_bar=True
        )
        # 保存最终模型（不带 .zip 后缀也会自动添加）
        model.save(f"{MODELS_DIR}/{MODEL_NAME}")
        print(">>> ✅ 训练完成！")

    except KeyboardInterrupt:
        # 允许用户按 Ctrl+C 随时暂停训练并保存当前模型
        print(">>> 🛑 训练中断，正在保存当前模型...")
        model.save(f"{MODELS_DIR}/interrupted_ppo")

    env.close()


import cv2  # 🔥 新增：导入 OpenCV 用于保存图片
import time  # 用于生成唯一文件名


# ... (前面的代码保持不变) ...

# ==========================================
# 🧪 推理/测试函数 (带死亡截图功能)
# ==========================================
def test():
    model_path = f"{MODELS_DIR}/best_model.zip"
    if not os.path.exists(model_path):
        model_path = f"{MODELS_DIR}/{MODEL_NAME}.zip"

    if not os.path.exists(model_path):
        print("❌ 没有找到模型文件！")
        return

    print(f">>> 🎮 正在加载模型: {model_path}")
    print(f">>> 📸 死亡截图模式已开启 (请保持名为 'Flappy Bird Replay' 的窗口在前台)")

    # 1. 依然保持 rgb_array，我们需要数据
    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=True)
    model = PPO.load(model_path)

    episodes = 10
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        last_frame = None

        while not done:
            # 2. 获取画面数据
            frame = env.render()
            last_frame = frame

            # 🔥🔥🔥 新增：手动把这个画面显示出来 🔥🔥🔥
            # OpenCV 需要 BGR 格式，Gym 是 RGB，转一下色
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 创建一个名为 "Flappy Bird Replay" 的窗口显示画面
            cv2.imshow("Flappy Bird Replay", bgr_frame)

            # 必须加这行！让窗口刷新 1ms，否则窗口会卡死或不显示
            # 按 'q' 可以提前退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                env.close()
                cv2.destroyAllWindows()
                return

            # AI 预测动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            score = info.get('score', score)

            # 3. 死亡保存截图
            if done:
                timestamp = int(time.time())
                file_name = f"{LOG_DIR}/death_ep{ep + 1}_score{score}_{timestamp}.png"
                if last_frame is not None:
                    # 使用刚才转换好的 bgr_frame 保存
                    cv2.imwrite(file_name, bgr_frame)
                    print(f"💀 第 {ep + 1} 局结束 (分:{score}) -> 截图已存: {file_name}")

    env.close()
    # 4. 跑完关掉窗口
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # 1. 训练模式 (第一次运行取消注释这个)
    # train()

    # 2. 观看模式 (训练完后查看效果)
    test()
# ==========================================
# 额外备注：
# - 本示例中导入了 DummyVecEnv、SubprocVecEnv、CheckpointCallback、Monitor，
#   它们在不同场景下会非常有用：
#     * DummyVecEnv / SubprocVecEnv: 用于更细粒度地控制并行策略（Subproc 在多核上更快，但进程间通信开销较高）
#     * CheckpointCallback: 可定期保存模型检查点（例如每 X 步保存一次）
#     * Monitor: 用来记录单个环境的 episode 数据（若使用自定义环境，可手动 wrap）
# - 如果需要我可以把注释翻译成英文、精简为 README 风格、或直接把超参数整理成一个可调 YAML/JSON 配置文件。
