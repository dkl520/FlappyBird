import os
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ================= 🚀 超参数配置 =================
ENV_ID = "FlappyBird-v0"
LEARNING_RATE = 2.5e-4
GAMMA = 0.99  # 折扣因子
LAMBDA = 0.95  # GAE 参数
EPS_CLIP = 0.2  # PPO 截断范围
K_EPOCHS = 5  # 每次更新循环次数 (SB3中通常是10，这里设4-10均可)
BATCH_SIZE = 64  # 小批量大小
UPDATE_TIMESTEP = 2048  # 每隔多少步更新一次网络 (对应 SB3 的 n_steps)
TOTAL_TIMESTEPS = 700_000
ENTROPY_COEF = 0.01  # 熵系数，鼓励探索

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📌 使用设备: {device}")


# ================= 🧠 1. 定义 Actor-Critic 网络 =================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # 共享特征提取层 (可选，也可以分开)
        # 激光雷达数据是 1D 向量，用 MLP 处理
        self.base_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        # Actor: 输出动作概率 (Logits)
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic: 输出状态价值 (Value)
        self.critic = nn.Linear(128, 1)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """用于在环境中采样动作"""
        x = self.base_layer(state)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(x)

        return action.item(), action_logprob.item(), state_val.item()

    def evaluate(self, state, action):
        """用于在更新时评估旧动作的概率和价值"""
        x = self.base_layer(state)

        action_probs = self.actor(x)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(x)

        return action_logprobs, state_values, dist_entropy


# ================= 🛠️ 2. 定义 PPO 算法逻辑 =================
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # 转换数据为 Tensor
        rewards = []
        discounted_reward = 0

        # --- 计算蒙特卡洛回报 (Returns) 或 GAE ---
        # 这里使用简单的 Cost-to-Go (Return) 计算，结合 GAE 效果更好，
        # 为了代码清晰，这里先计算 Reward-to-Go用于计算优势
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 归一化回报 (这对收敛很关键)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 将 list 转为 tensor
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(memory.state_values, dim=0)).detach().to(device)

        # 计算优势函数 (Advantage) = Return - Value
        # 在标准的 PPO 中通常使用 GAE，这里简化为 returns - old_values
        advantages = rewards.detach() - old_state_values.detach()

        # --- PPO 更新循环 (K epochs) ---
        for _ in range(K_EPOCHS):
            # 评估旧状态和动作
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            # 计算比率 ratio (pi_theta / pi_theta_old)
            # exp(log_prob - old_log_prob) = prob / old_prob
            ratio = torch.exp(logprobs - old_logprobs)

            # --- 核心 Loss 公式 ---
            # 1. Surrogate Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            loss_actor = -torch.min(surr1, surr2)

            # 2. Value Loss (MSE)
            loss_critic = self.MseLoss(state_values, rewards)

            # 3. Total Loss (加上熵正则项鼓励探索)
            loss = loss_actor + 0.5 * loss_critic - ENTROPY_COEF * dist_entropy

            # 反向传播
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 更新旧策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=device))


# ================= 📦 3. 简单的经验回放缓冲区 =================
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.state_values[:]


# ================= 🏃 训练流程 =================
def train():
    print("🚀 开始手写 PPO 训练...")

    # 创建环境
    env = gym.make(ENV_ID, use_lidar=True, background=None)

    # 获取维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    memory = Memory()
    ppo_agent = PPO(state_dim, action_dim)

    time_step = 0
    i_episode = 0

    while time_step < TOTAL_TIMESTEPS:
        state, _ = env.reset()
        current_ep_reward = 0
        done = False

        while not done:
            # 1. 选择动作
            # 注意：state 需要转为 tensor 且增加 batch 维度
            state_tensor = torch.FloatTensor(state).to(device)
            action, logprob, val = ppo_agent.policy_old.act(state_tensor)

            # 2. 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 3. 存储数据到 Buffer
            memory.states.append(state_tensor)
            memory.actions.append(torch.tensor(action).to(device))
            memory.logprobs.append(torch.tensor(logprob).to(device))
            memory.state_values.append(torch.tensor(val).to(device))
            memory.rewards.append(reward)
            memory.is_terminals.append(terminated)  # 注意这里用 terminated 比较好

            state = next_state
            current_ep_reward += reward
            time_step += 1

            # 4. 如果达到了更新步数，进行 PPO 更新
            if time_step % UPDATE_TIMESTEP == 0:
                print(f"🔄 Step {time_step}: 更新策略网络...")
                ppo_agent.update(memory)
                memory.clear()

            if done:
                break

        i_episode += 1

        # 简单打印日志
        if i_episode % 20 == 0:
            print(
                f"Episode: {i_episode} \t Timestep: {time_step} \t Reward: {current_ep_reward:.2f} \t Score: {info.get('score', 0)}")

        # 定期保存
        if time_step % 50000 == 0:
            os.makedirs("manual_models", exist_ok=True)
            ppo_agent.save(f"manual_models/ppo_flappy_{time_step}.pth")

    print("✅ 训练结束")
    ppo_agent.save("manual_models/ppo_flappy_final.pth")
    env.close()


# ================= 🎮 测试流程 =================
def test():
    print("👀 加载模型进行测试...")
    env = gym.make(ENV_ID, render_mode="human", use_lidar=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo_agent = PPO(state_dim, action_dim)
    model_path = "manual_models/ppo_flappy_final.pth"

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}，请先训练。")
        return

    ppo_agent.load(model_path)

    for ep in range(5):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            # 测试时取确定性动作：选概率最大的
            # 但手写act函数通常是采样的。为了演示方便，这里我们还是用act采样，
            # 真正严谨的测试应该取 actor 输出 logits 最大的那个 index。
            with torch.no_grad():
                action_probs = ppo_agent.policy.actor(state_tensor)
                action = torch.argmax(action_probs).item()  # 贪婪策略

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score = info.get('score', 0)

        print(f"Episode {ep + 1} Score: {score}")

    env.close()


if __name__ == "__main__":
    # 切换这里来训练或测试
    train()
    # test()