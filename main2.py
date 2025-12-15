import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor  # 用于监控
import os
import glob
import warnings
import torch

# 屏蔽烦人的 gymnasium 警告
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# ==========================================
# ⚙️ 全局配置
# ==========================================
MODELS_DIR = "models/flappy_bird"
LOG_DIR = "logs/flappy_bird"
MODEL_NAME = "flappy_bird_final"

# ✅ 保持你的成功经验：使用 12 维简单状态
USE_LIDAR = False

# 🔴 画面开关
SHOW_GAME = False

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# --- 优化1: 动态学习率调度器 ---
def linear_schedule(initial_value: float):
    """
    学习率从 initial_value 线性下降到 0
    """

    def func(progress_remaining: float):
        return progress_remaining * initial_value

    return func


def train():
    print(f">>> [训练] 初始化环境 (Lidar={USE_LIDAR})...")
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=USE_LIDAR)
    # 包装环境以记录更详细的数据
    env = Monitor(env, LOG_DIR)

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # 优化：每10万步存一次，不用太频繁
        save_path=MODELS_DIR,
        name_prefix="ckpt"
    )

    # 1. 尝试加载已有模型（断点续练）
    final_path = f"{MODELS_DIR}/{MODEL_NAME}.zip"
    if os.path.exists(final_path):
        print(f">>> ♻️ 检测到已有模型 {MODEL_NAME}，正在加载并【继续训练】...")
        model = DQN.load(final_path, env=env, tensorboard_log=LOG_DIR)

        # ⬇️ 关键修改：加载后手动关闭 verbose，防止刷屏表格
        model.verbose = 0

        # 续练时的参数微调
        model.exploration_initial_eps = 0.05
        model.exploration_final_eps = 0.01
    else:
        print(">>> 🆕 未找到旧模型，开始【从零训练】...")
        # 优化2:
        #      网络结构微调
        # [512, 512] 对这个游戏可能有点过拟合，[256, 256] 通常更稳且快
        # 但既然你用 512 效果好，我们保持深度，但稍微优化宽度
        policy_kwargs = dict(net_arch=[512, 512])

        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,  # ⬇️ 关键修改：设置为 0 以关闭控制台的表格输出
            tensorboard_log=LOG_DIR,

            # 优化3: 使用动态学习率 (开始快，后来慢)
            learning_rate=linear_schedule(1e-4),

            buffer_size=500_000,  # 50万够了，100万可能占太多内存
            learning_starts=10000,
            batch_size=64,  # 优化：128 通常比 64 更稳
            gamma=0.99,

            # 优化4: 训练频率调整 (玩4步，学1次) -> 标准DQN设置
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,  # 更频繁地更新目标网络

            exploration_fraction=0.2,  # 前20%的时间都在探索
            exploration_final_eps=0.01,
            policy_kwargs=policy_kwargs,
        )
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,  # ⬇️ 关键修改：设置为 0 以关闭控制台的表格输出
            tensorboard_log=LOG_DIR,
            learning_rate=1e-4,
            buffer_size=1000000,  # Optimized: Increase to default 1M for more diverse experiences
            learning_starts=10000,
            batch_size=64,  # Optimized: Balance between 32 (default) and your 128 for stability
            gamma=0.99,
            exploration_fraction=0.15,  # Optimized: Slightly longer exploration phase
            exploration_final_eps=0.02,  # Optimized: Higher final eps to avoid premature convergence
            policy_kwargs=policy_kwargs,
            target_update_interval=5000  # Optimized: More frequent updates for stability (default 10k)
        )

    # 优化5: 增加步数
    # 30万步(3_000_00)太少，100万步能让它从"新手"变成"大师"
    TRAIN_STEPS = 500_000
    print(f">>> [训练] 目标步数: +{TRAIN_STEPS} 步...")
    print(f">>> [提示] 表格已隐藏，请观察下方的进度条...")

    try:
        model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True, callback=checkpoint_callback)
        model.save(f"{MODELS_DIR}/{MODEL_NAME}")
        print(f">>> [训练] 完成并保存！")
    except KeyboardInterrupt:
        print("\n>>> [中断] 正在保存紧急备份...")
        model.save(f"{MODELS_DIR}/interrupted_model")
        print(">>> 保存成功。")

    env.close()


def find_latest_model():
    final_path = f"{MODELS_DIR}/{MODEL_NAME}.zip"
    if os.path.exists(final_path):
        return final_path

    print(f">>> 未找到最终模型，寻找最近的 Checkpoint...")
    list_of_files = glob.glob(f"{MODELS_DIR}/ckpt_*.zip")
    if not list_of_files: return None
    return max(list_of_files, key=os.path.getctime)


def test():
    load_path = find_latest_model()
    if not load_path:
        print(">>> ❌ 没找到模型，请先运行 train()！")
        return

    print(f">>> [测试] 加载模型: {load_path}")

    render_mode = "human" if SHOW_GAME else None

    # 保持 use_lidar=False
    env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=USE_LIDAR)
    model = DQN.load(load_path)

    scores = []
    print(">>> [测试] 开始 10 局测试...")
    for ep in range(10):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score = info.get("score", 0)

        scores.append(score)
        print(f"   第 {ep + 1} 局得分: {score}")

    avg = sum(scores) / len(scores)
    print(f"\n>>> [总结] 平均分: {avg}")
    if avg < 10:
        print(">>> 💡 建议：既然能跑 10 分了，说明路子对了！继续 train() 更多步数，分数会指数级上涨。")

    env.close()


if __name__ == "__main__":
    # 1. 既然你的方向是对的，建议再狠狠训练它 100万步
    # train()

    # 2. 测试时记得保持 SHOW_GAME = True 来看它表演
    test()
