import os
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm  # 导入进度条库

# ================= 🚀 超参数配置 =================
ENV_ID = "FlappyBird-v0"
LEARNING_RATE = 2.5e-4
GAMMA = 0.99  # 折扣因子
LAMBDA = 0.95  # GAE 参数
EPS_CLIP = 0.2  # PPO 截断范围
K_EPOCHS = 10  # 更新循环次数
BATCH_SIZE = 64  # 小批量大小
UPDATE_TIMESTEP = 2048  # 收集多少步数据更新一次
TOTAL_TIMESTEPS = 1_000_000
ENTROPY_COEF = 0.01

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"  # 保持你原来的设置
print(f"📌 使用设备: {device}")


# ================= 🛡️ 安全奖励包装器 =================
class StrictSafetyWrapper(gym.Wrapper):
    def __init__(self, env, safe_dist=0.10):
        super().__init__(env)
        self.safe_dist = safe_dist

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        min_dist = np.min(obs)

        # 惩罚贴管飞行 (负反馈)
        if min_dist < self.safe_dist:
            reward -= 0.03

        return obs, reward, terminated, truncated, info


# ================= 🧠 1. 定义 Actor-Critic 网络 =================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # 共享特征提取层
        self.base_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic
        self.critic = nn.Linear(128, 1)

    def act(self, state):
        x = self.base_layer(state)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(x)

        return action.item(), action_logprob.item(), state_val.item()

    def evaluate(self, state, action):
        x = self.base_layer(state)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(x)

        return action_logprobs, state_values, dist_entropy


# ================= 🛠️ 2. PPO 算法逻辑 =================
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        old_states = torch.stack(memory.states).detach().to(device)
        old_actions = torch.stack(memory.actions).detach().to(device)
        old_logprobs = torch.stack(memory.logprobs).detach().to(device)
        old_state_values = torch.stack(memory.state_values).detach().to(device).squeeze()

        rewards = memory.rewards
        is_terminals = memory.is_terminals
        advantages = []
        gae = 0

        # GAE 计算
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = 0
            else:
                next_val = old_state_values[i + 1].item()
            curr_val = old_state_values[i].item()
            mask = 1 - is_terminals[i]
            delta = rewards[i] + GAMMA * next_val * mask - curr_val
            gae = delta + GAMMA * LAMBDA * mask * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = advantages + old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        dataset_size = old_states.size(0)

        # PPO 更新
        for _ in range(K_EPOCHS):
            for index in range(0, dataset_size, BATCH_SIZE):
                batch_indices = slice(index, min(index + BATCH_SIZE, dataset_size))
                batch_states = old_states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                logprobs, state_values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)
                state_values = torch.squeeze(state_values)
                ratio = torch.exp(logprobs - batch_logprobs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * batch_advantages
                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = self.MseLoss(state_values, batch_returns)
                loss = loss_actor + 0.5 * loss_critic - ENTROPY_COEF * dist_entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # 加上 weights_only=True (如果你的pytorch版本较新) 或者忽略它
        # 这里为了兼容性，通常可以保持原样，或者显式加上 weights_only=False 消除歧义
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.policy_old.load_state_dict(state_dict)
        self.policy.load_state_dict(state_dict)

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


# ================= 🏃 训练流程 (带断点续训 + 彩色进度条) =================
def train():
    print("🚀 准备开始训练...")

    env = gym.make(ENV_ID, use_lidar=True, background=None)
    env = StrictSafetyWrapper(env, safe_dist=0.09)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    memory = Memory()
    ppo_agent = PPO(state_dim, action_dim)

    # ------------------- 🔄 断点续训逻辑 -------------------
    model_dir = "manual_models"
    final_model_name = "ppo_flappy_final.pth"
    resume_path = os.path.join(model_dir, final_model_name)

    if os.path.exists(resume_path):
        print(f"🔄 发现上次训练模型: {resume_path}")
        try:
            ppo_agent.load(resume_path)
            print("✅ 模型加载成功！将在该模型基础上继续训练 (Resume Training)")
        except Exception as e:
            print(f"⚠️ 模型加载出错 ({e})，将从头开始训练。")
    else:
        print("🆕 未找到已有模型，将从头开始训练 (Start From Scratch)")
    # ------------------------------------------------------

    time_step = 0
    running_reward = 0  # 用于计算平滑平均分

    # 🟢 初始化 tqdm 进度条 (colour='green' 实现彩色效果)
    pbar = tqdm(total=TOTAL_TIMESTEPS, desc="Training", unit="step", colour='green')

    while time_step < TOTAL_TIMESTEPS:
        state, _ = env.reset()
        current_ep_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            action, logprob, val = ppo_agent.policy_old.act(state_tensor)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            memory.states.append(state_tensor)
            memory.actions.append(torch.tensor(action).to(device))
            memory.logprobs.append(torch.tensor(logprob).to(device))
            memory.state_values.append(torch.tensor(val).to(device))
            memory.rewards.append(reward)
            memory.is_terminals.append(terminated)

            state = next_state
            current_ep_reward += reward
            time_step += 1

            # 🟢 更新进度条
            pbar.update(1)

            # PPO 更新
            if time_step % UPDATE_TIMESTEP == 0:
                ppo_agent.update(memory)
                memory.clear()

            if done:
                break

        # 🟢 计算平滑平均分
        if running_reward == 0:
            running_reward = current_ep_reward
        else:
            running_reward = 0.05 * current_ep_reward + 0.95 * running_reward

        # 🟢 设置进度条后缀
        pbar.set_postfix({
            'Last': f'{current_ep_reward:.2f}',
            'Avg': f'{running_reward:.2f}'
        })

        # ================= 💾 定期保存 =================
        if time_step % 50000 == 0:
            os.makedirs(model_dir, exist_ok=True)  # 确保文件夹存在
            save_path = os.path.join(model_dir, f"ppo_flappy_{time_step}.pth")

            ppo_agent.save(save_path)
            abs_path = os.path.abspath(save_path)
            pbar.write(f"💾 阶段保存: {abs_path}")

    pbar.close()  # 关闭进度条

    # ================= 💾 最终保存 =================
    try:
        os.makedirs(model_dir, exist_ok=True)  # 再次确保文件夹存在
        final_save_path = os.path.join(model_dir, final_model_name)

        ppo_agent.save(final_save_path)

        print("✅ 训练结束")
        print(f"📍 最终模型位置: {os.path.abspath(final_save_path)}")

    except Exception as e:
        print(f"❌ 最终保存失败: {e}")

    env.close()

def test():
    print("👀 加载模型进行测试 (无尽模式)...")
    print("💡 提示：按 Ctrl+C 可以强制停止程序")

    # 1. 设置极大的步数限制 (1亿步)，确保不会因为超时而重置
    env = gym.make(ENV_ID, render_mode="human", use_lidar=True, background=None, max_episode_steps=100000000)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo_agent = PPO(state_dim, action_dim)
    model_path = "manual_models/ppo_flappy_final.pth"

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}，请先运行 train() 进行训练。")
        return

    ppo_agent.load(model_path)

    episode_cnt = 0

    # 2. 改回 while True，实现真正的“无限局数”
    while True:
        episode_cnt += 1
        state, _ = env.reset()
        terminated = False
        truncated = False  # 初始化 truncated
        score = 0
        step_cnt = 0

        # 只要没死，就一直飞 (忽略 truncated，除非你真的想看它飞一亿步)
        while not terminated:
            step_cnt += 1
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                features = ppo_agent.policy.base_layer(state_tensor)
                action_probs = ppo_agent.policy.actor(features)
                action = torch.argmax(action_probs).item()

            state, reward, terminated, truncated, info = env.step(action)
            score = info.get('score', 0)

            # 如果触发了 truncated (虽然设置了1亿步不太可能)，我们强行让它不要停
            # 注意：如果环境内部有硬性时间限制，这里可能会出警告，但通常有效
            if truncated:
                # 打印一下看看是不是真的超时了
                print("超时了！！！！！")
                pass

                # 🛑 游戏结束，打印原因
        print(f"Episode {episode_cnt} | Score: {score} | Steps: {step_cnt}")
        if terminated:
            print(f"   💀 死亡原因: Terminated (判定死亡，可能是撞柱子、掉地或 **撞天花板**)")
        elif truncated:
            print(f"   ⏳ 结束原因: Truncated (超时强制结束)")

        # 暂停 1 秒让你看清最后的画面
        import time
        time.sleep(10)

    env.close()
if __name__ == "__main__":
    # train()  # 训练模式
    test()   # 测试模式