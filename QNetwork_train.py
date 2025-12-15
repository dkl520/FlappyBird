# 文件名: train.py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ==============================
# QNetwork: 用于近似 Q(s, a) 的神经网络
# ==============================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        state_dim: 状态向量的维度（输入大小）
        action_dim: 动作数量（输出大小）
        网络结构：全连接 128 -> ReLU -> 128 -> ReLU -> action_dim
        这里使用小型 MLP 作为 function approximator，方便在 CartPole 上快速训练。
        """
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),  # 全连接层：输入 -> 128
            nn.ReLU(),                  # 激活函数 ReLU
            nn.Linear(128, 128),        # 全连接层：128 -> 128
            nn.ReLU(),                  # 激活函数 ReLU
            nn.Linear(128, action_dim)  # 输出层：128 -> 动作数（Q 值）
        )

    def forward(self, x):
        """
        前向传播：输入 x（通常为状态），输出每个动作对应的 Q 值（未归一化）。
        注意：x 应该是 FloatTensor，且形状为 [batch, state_dim]（若单样本需 unsqueeze）。
        """
        return self.fc(x)


# ==============================
# ReplayBuffer: 经验回放缓冲区（固定容量的循环队列）
# ==============================
class ReplayBuffer:
    def __init__(self, capacity):
        """
        capacity: 缓冲区最大样本数
        使用 collections.deque(maxlen=capacity) 来自动丢弃最早的样本。
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        将一个 transition (s, a, r, s', done) 添加到缓冲区。
        注意：这里保存原始 numpy/list 状态；在 sample 时再转换为 tensor。
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        随机采样 batch_size 个样本并打包返回。
        返回：state, action, reward, next_state, done（每个是 tuple 列表）
        使用 random.sample 做无放回抽样，保证训练稳定性。
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        """
        返回当前缓冲区中样本数量（用于判断是否可进行一次训练更新）。
        """
        return len(self.buffer)


# ==============================
# DQNAgent: 包含 policy 网络、target 网络、优化器与学习逻辑
# ==============================

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        """
        初始化 DQN Agent 的各种组件与超参数：
        - 两个网络（policy_net, target_net）：target_net 用于稳定目标 Q 估计（固定一段时间再同步）
        - 优化器：Adam（学习率 1e-3）
        - ReplayBuffer：容量 10000
        - 训练相关超参：batch_size, gamma, epsilon 贪心策略参数等
        - device: 若有 GPU 则使用 CUDA，否则使用 CPU
        """
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 策略网络：用于选择动作并训练
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        # 目标网络：用于计算目标 Q 值，参数来源于 policy_net 的滞后拷贝
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        # 将 target_net 初始化为 policy_net 的参数（完全相同）
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # target_net 在评估目标时保持 eval 模式（不启用 dropout/batchnorm 之类）
        self.target_net.eval()
        # 把这个神经网络切换到“考试模式”（评估模式）。

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

        # 经验回放
        self.memory = ReplayBuffer(10000)
        # 训练相关超参数
        self.batch_size = 64
        self.gamma = 0.99  # 折扣因子
        # epsilon-greedy 策略参数（用于探索）
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # target network 同步频率（以 step/episode 计）：这里在训练循环里每 5 episode 同步一次，
        # 但 agent 里也保留此参数以便修改策略（当前代码中未直接使用这个字段）
        self.target_update_freq = 100

    def act(self, state):
        """
        基于 epsilon-greedy 策略选择动作：
        - 以概率 epsilon 随机选动作（探索）
        - 否则将 state 输入 policy_net，选择 Q 最大的动作（利用）
        state: 可以是 numpy 数组或 list（单个状态）
        返回：动作索引（int）
        """
        # 随机探索

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # 将状态转换为 Tensor 并移动到 device，上面加了一个 batch 维度
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state)  # shape: [1, action_dim]
        # 返回 Q 值最大的动作索引
        return q_values.argmax().item()

    def update(self):
        """
        执行一次网络参数更新（学习一步）：
        - 从 replay buffer 中采样 batch
        - 计算当前 Q(s,a)（由 policy_net 给出）
        - 计算目标 Q 值：r + gamma * max_a' target_net(s', a')（若 done 则不包含后续值）
        - 最小化 MSE loss(current_q, target_q)
        - 反向传播并更新 policy_net 参数
        - 衰减 epsilon（逐渐减少探索）
        注意：若 replay buffer 中样本少于 batch_size 则不执行更新
        """
        # 若经验不足，则跳过更新
        if len(self.memory) < self.batch_size:
            return

        # 采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 将采样到的数据转换为 Tensor 并移动到 device
        # states / next_states 可能是 list of np.arrays，使用 np.array 打包后转换
        states = torch.FloatTensor(np.array(states)).to(self.device)         # shape: [B, state_dim]
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)    # shape: [B, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)   # shape: [B, 1]
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)  # shape: [B, state_dim]
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)       # shape: [B, 1]

        # 计算当前 Q 值：policy_net(states) -> gather 对应动作
        current_q = self.policy_net(states).gather(1, actions)  # shape: [B, 1]

        # 使用 target_net 计算下一状态的最大 Q（不对 target 网络进行梯度计算）
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)  # shape: [B, 1]
            # 若 done 为 1（回合结束），则不加上后续值
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # 均方误差损失
        loss = nn.MSELoss()(current_q, target_q)

        # 优化步骤
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon 衰减，保持在 epsilon_min 以上
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ==============================
# 训练主逻辑（train 函数）
# ==============================
def train():
    """
    训练流程：
    1. 创建环境（CartPole-v1）
    2. 初始化 agent（包括网络、replay buffer 等）
    3. 进行若干 episode：
       - 每个 episode 从 env.reset() 开始，不断与环境交互并存储 transition
       - 每一步调用 agent.update()（只在 replay buffer 足够时真正更新）
       - 每隔若干 episode 同步 target 网络参数
    4. 达到某一分数阈值结束训练并保存模型参数
    """
    print(">>> 开始训练...")
    env = gym.make("CartPole-v1")  # 创建 CartPole 环境
    # 获取状态维度与动作数（适配不同环境）
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    episodes = 500  # 最多训练的 episode 数
    for episode in range(episodes):
        # reset 返回 (obs, info)（Gymnasium 风格）
        state, _ = env.reset()
        total_reward = 0

        # 单个 episode 的交互循环
        while True:
            # 根据当前状态选择动作（epsilon-greedy）
            action = agent.act(state)
            # 与环境交互，得到下一状态、奖励、是否终止等（Gymnasium API）
            next_state, reward, terminated, truncated, _ = env.step(action)
            # done 为 True 表示回合结束（终止或被截断）
            done = terminated or truncated

            # 这里对 terminated（被环境判定失败的情况）进行额外的惩罚：
            # 如果 terminated==True 表示失败（比如 CartPole 摔倒），给 -10 奖励，便于学习更稳定的策略。
            if terminated:
                reward = -10

            # 将 transition 存入回放缓冲区
            agent.memory.push(state, action, reward, next_state, done)

            # 状态前进
            state = next_state
            total_reward += reward

            # 每一步尝试学习（当 replay buffer 中样本足够时，会真正执行参数更新）
            agent.update()

            # 如果回合结束，则跳出循环
            if done:
                break

        # 每隔若干 episode 将 policy_net 的参数拷贝到 target_net 中以稳定训练
        # 这里使用 (episode % 5 == 0) 的策略：即每 5 个 episode 同步一次
        if episode % 5 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # 每 20 个 episode 打印一次分数（便于观察训练进度）
        if (episode + 1) % 20 == 0:
            print(f"Episode: {episode + 1}, Score: {total_reward}")

        # 如果在某个 episode 中得到极高分数（>=500）则认为已解决任务并提前终止训练
        # 注意：CartPole-v1 的默认最大 step 是 500，满分通常是 500。
        if total_reward >= 500:
            print(f"Solved at episode {episode + 1}!")
            break

    env.close()

    # ==============================
    # 保存训练好的模型参数
    # ==============================
    # 只保存 policy_net 的参数（state_dict），文件名: dqn_cartpole.pth
    # 这样后续可以通过 model.load_state_dict(torch.load("dqn_cartpole.pth")) 恢复
    torch.save(agent.policy_net.state_dict(), "dqn_cartpole.pth")
    print(">>> 模型已保存为 'dqn_cartpole.pth'")

# 当以脚本形式运行时，调用 train()
if __name__ == "__main__":
    train()
