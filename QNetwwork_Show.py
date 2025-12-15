# 文件名: demo.py
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np


# ==========================================================
# 1. QNetwork 神经网络结构
# ⚠ 必须与 train.py 中的完全一致，否则加载权重会报错
# ==========================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        # 全连接前馈神经网络，结构和训练时保持一致
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),  # 第一层：输入 -> 128
            nn.ReLU(),
            nn.Linear(128, 128),        # 第二层：128 -> 128
            nn.ReLU(),
            nn.Linear(128, action_dim)  # 输出层：动作数
        )

    def forward(self, x):
        # 前向传播
        return self.fc(x)


# ==========================================================
# 2. 游戏演示函数
# ==========================================================
def show():
    # 创建游戏环境，开启渲染窗口
    # render_mode="human" 表示实时显示画面
    env = gym.make("CartPole-v1", render_mode="human")

    # 环境状态维度，例如 CartPole 是 4 维
    state_dim = env.observation_space.shape[0]
    # 动作数量，例如 CartPole 是 2（左 / 右）
    action_dim = env.action_space.n

    # 使用 GPU 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型实例（结构要和训练时一致）
    model = QNetwork(state_dim, action_dim).to(device)

    # ======================================================
    # 加载训练好的模型参数
    # ======================================================
    print(">>> 正在加载模型 'dqn_cartpole.pth' ...")
    try:
        # map_location=device：即使训练在 GPU，也能在 CPU 上加载
        model.load_state_dict(
            torch.load("dqn_cartpole.pth", map_location=device, weights_only=True)
        )
        model.eval()  # 切换到推理模式（不启用 dropout 等）
    except FileNotFoundError:
        print("错误：找不到 'dqn_cartpole.pth' 文件。请先运行 train.py！")
        return

    # ======================================================
    # 5 局游戏演示
    # ======================================================
    print(">>> 开始演示 (按 Ctrl+C 退出)")
    for i in range(5):
        # 重置环境，得到初始状态
        state, _ = env.reset()
        total_reward = 0

        # 一直让模型控制直到游戏结束
        while True:
            # 转 tensor，并添加 batch 维度 (1, state_dim)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # 推理，不计算梯度
            with torch.no_grad():
                q_values = model(state_tensor)  # 得到每个动作的 Q 值
                action = q_values.argmax().item()  # 选择 Q 值最大的动作

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 更新状态
            state = next_state
            total_reward += reward

            # 环境结束：打印得分
            if done:
                print(f"演示局 {i + 1} 得分: {total_reward}")
                break

    # 关闭环境
    env.close()


# ======================================================
# 程序入口
# ======================================================
if __name__ == "__main__":
    show()
