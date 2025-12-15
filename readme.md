# Flappy Bird 强化学习 / Flappy Bird Reinforcement Learning

本项目使用 `Gymnasium` 的 `FlappyBird-v0` 环境和两套强化学习方案（`Stable-Baselines3` 的 DQN 以及自定义的 PPO）训练智能体自动玩 Flappy Bird。你可以选择显示游戏画面进行测试，也可以在无渲染模式下进行高效训练。

This project trains an agent to play Flappy Bird using the `Gymnasium` `FlappyBird-v0` environment with two RL approaches: DQN from `Stable-Baselines3` and a custom PPO implementation. You can test with rendering enabled or train efficiently without rendering.

## 目录概览 / Directory Overview
- `main.py` / `main2.py`: 使用 Stable-Baselines3 DQN 训练与测试（含断点续训、检查点保存）
- `v12_最终版本.py`: 自定义 PPO 训练与“无限测试”模式
- `manual_models/`: 手动保存的 PPO `.pth` 模型
- `trained_models/`: 历史训练得到的 `.zip` 模型集合
- `.idea/`: IDE 项目配置
- 其他 `v*` 脚本：不同实验版本（如无限模式、中文提示等）

## 环境与依赖 / Environment & Dependencies
- Python 3.9+ 建议（Recommendation）
- 必需（Required）:
  - `gymnasium`, `flappy-bird-gymnasium`
  - `stable-baselines3`
  - `numpy`, `tqdm`
  - `torch`（可选 GPU 加速）
- GPU 加速安装（CUDA 12.1）/ GPU install (CUDA 12.1):
  - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
  - 如无 GPU，可直接 `pip install torch torchvision torchaudio`

## 安装 / Installation
```bash
pip install gymnasium flappy-bird-gymnasium stable-baselines3 numpy tqdm
# GPU 可选（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
验证 GPU / Check GPU:
```bash
python test_gpu.py
```

## 快速开始 / Quick Start
### 方案一：DQN（Stable-Baselines3）
- 训练（Train）：编辑 `main.py`，取消 `train()` 调用，然后运行：
```bash
python main.py
```
- 测试（Test）：确保 `SHOW_GAME = True`，默认执行 `test()`：
```bash
python main.py
```
- 模型保存（Model Save）：`models/flappy_bird/flappy_bird_final.zip` 与周期性 `ckpt_*.zip`

关键代码位置 / Key references:
- 训练入口：`main.py:43`（`train()`）
- 测试入口：`main.py:128`（`test()`）
- 渲染控制：`main.py:24-26`（`SHOW_GAME`）

### 方案二：PPO（自定义实现 / Custom）
- 训练（Train）：编辑 `v12_最终版本.py`，启用 `train()` 调用后运行：
```bash
python v12_最终版本.py
```
- 测试（Test/无限模式）：默认执行 `test()`，需先准备模型：
  - 模型路径：`manual_models/ppo_flappy_final12.pth`
```bash
python v12_最终版本.py
```
关键代码位置 / Key references:
- 训练入口：`v12_最终版本.py:184`（`train()`）
- 测试入口：`v12_最终版本.py:361-363`（`test()`）
- 安全奖励包装器：`v12_最终版本.py:28-43`（`StrictSafetyWrapper`）

## 常用配置 / Common Configuration
- 是否使用 Lidar 状态（12 维简化状态）：`main.py:21-23`（`USE_LIDAR`）
- 可视化开关（渲染模式）：`main.py:24-26`（`SHOW_GAME` 与 `render_mode`）
- 训练步数与检查点频率：`main.py:99-107`、`main.py:49-53`
- PPO 超参数与进度条：`v12_最终版本.py:11-22`、`v12_最终版本.py:215-263`

## 运行提示 / Tips
- 首次使用 DQN 时，若不存在最终模型，将自动从零开始训练并定期保存检查点。
- PPO 测试为“无限模式”，如需停止，使用 `Ctrl+C`。
- 若遇到导入失败，请确认已安装 `flappy-bird-gymnasium` 与兼容版本的 `gymnasium`。
- 若需更高分数，增加训练步数（例如 DQN 的 `TRAIN_STEPS`）。

## 致谢 / Acknowledgements
- `Gymnasium` 与 `flappy-bird-gymnasium` 提供环境支持
- `Stable-Baselines3` 提供 DQN 算法实现
- `PyTorch` 提供深度学习计算后端
