import gymnasium as gym
import flappy_bird_gymnasium
import time
import numpy as np


def test_new_lidar_settings():
    print("🔍 开始测试新的雷达设置 (240度 / 240线)...")

    # 初始化环境，开启 render_mode="human" 以便肉眼观察
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)

    # ================= 1. 验证容器定义 (Observation Space) =================
    obs_shape = env.observation_space.shape
    print(f"\n📊 检查点 1: 环境定义的观察空间形状")
    print(f"   -> 你的环境声明它需要: {obs_shape}")

    if obs_shape == (240,):
        print("   ✅ [通过] 容器定义已成功改为 240！")
    else:
        print(f"   ❌ [失败] 容器定义仍为 {obs_shape}。")
        print("      请检查 flappy_bird_env.py 中的 observation_space 设置。")
        env.close()
        return

    # ================= 2. 验证雷达逻辑 (Actual Output) =================
    obs, info = env.reset()
    real_data_len = len(obs)
    print(f"\n📡 检查点 2: 实际生成的雷达数据")
    print(f"   -> 实际返回的数据长度: {real_data_len}")

    if real_data_len == 240:
        print("   ✅ [通过] 雷达逻辑已成功修改，正在输出 240 个数据点！")
    else:
        print(f"   ❌ [失败] 雷达逻辑未生效，实际输出长度为 {real_data_len}。")
        print("      请检查 flappy_bird_gymnasium/envs/lidar.py 中的 scan 函数。")
        env.close()
        return

    # ================= 3. 视觉验证 =================
    print("\n👀 [视觉检查] 请看弹出的游戏窗口：")
    print("   1. 你应该看到红色的激光线非常密集。")
    print("   2. 视野应该非常宽 (240度)，甚至能看到后面一点点 (超过了180度的平角)。")
    print("   3. 程序将运行 1000 步演示 (随机动作)...")

    for _ in range(1000):
        # 随机动作，只是为了让画面动起来
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # 再次确保每一帧的数据都是 240
        assert len(obs) == 240, "数据长度在运行中发生变化！"

        if terminated or truncated:
            env.reset()

        # 稍微加点延迟，让你看清楚雷达线的范围
        time.sleep(0.03)

    env.close()
    print("\n✨ 测试完成！如果以上都打钩，你可以开始重新训练你的 240度 模型了。")


if __name__ == "__main__":
    test_new_lidar_settings()