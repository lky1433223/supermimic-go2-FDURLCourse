# RL-Go2

强化学习机械狗作业：Genesis 训练，MuJoCo 做 sim2sim 验证。当前是框架及平地速度跟踪初始任务，尚未训练出行走策略。

## 环境与资产

```bash
uv sync --extra train --locked
uv run --extra train python scripts/fetch_go2.py  # 已有资产时无需重复执行
uv run --extra train python scripts/check_env.py
```

Python 固定 3.12，依赖锁定在 `uv.lock`，PyCharm 解释器选 `.venv/bin/python`。
纯配置解析/MuJoCo Python 库可用 `uv sync --locked` 安装；本项目的共享控制、任务、PPO 与策略评估均需 `--extra train` 中的 PyTorch。

Go2 来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2)，
下载器固定提交版本；原始 `LICENSE` 和包含文件哈希的 `source.json` 保存在 `assets/robots/go2/`。
Genesis 与 MuJoCo 读取同一份未改写的机器人 MJCF 和网格。

## 框架

```text
configs/scenes/       场景：实体、资产、位姿和物理步长
configs/robots/       标准关节顺序、PD、力矩限制和控制周期
configs/tasks/        速度命令、奖励、终止条件
configs/experiments/  组合配置、设备、并行环境数量、算法和课程选择
src/rl_go2/
  config.py          YAML 加载与当前支持范围校验
  state.py           统一批量状态、坐标变换
  backends/          PhysicsBackend 协议、Genesis 和 MuJoCo 实现
  control.py         两后端共用的逐物理步 PD 控制
  tasks.py           平地速度跟踪任务
  env.py             控制步、重置、统计及可选状态转移记录
  curriculum.py      课程接口；当前只实现固定难度
  trainers/ppo.py    RSL-RL VecEnv 适配、训练和策略导出
  trainers/supermimic.py  世界模型接口，训练明确报 NotImplementedError
  cli.py             配置检查、smoke、训练、策略评估
```

场景采用 Unisis 风格子集，schema 为本项目的 `rl-go2.scene/v1`，不声明完全兼容。
目前支持一个浮动基座 MJCF 机器人、静态 plane/box；传感器、mesh/USD、动态道具和 Isaac Sim 尚未实现，遇到时明确报错。
box 的 `size` 是三条完整边长。单位为米/秒/弧度，右手系 Z-up，四元数为 wxyz。
资产路径相对于场景 YAML 解析，实验引用的配置路径相对于实验 YAML 解析。
`dt` 是一次 backend.step 的物理时间；控制周期为 `dt * control_decimation`，默认 0.005 × 4 = 0.02 秒。
Genesis 内部 substeps 固定为 1，避免控制周期被隐式改变。

共享策略观测为 45 维：机身坐标角速度(3)、投影重力(3)、速度命令(3)、默认角度偏移(12)、关节速度(12)、上一步裁剪动作(12)。
动作经裁剪与缩放后得到默认关节角的偏移，逐物理步执行共享 PD 并限制力矩。当前不模拟执行延迟。
MuJoCo 初期只实现 CPU 单环境评估；Genesis 支持批量环境。设备选择 cpu/cuda/mps，实际训练设备与驱动需在训练前验证。

`Transition` 接口记录重置前的状态转移，单独保留 terminated/truncated。
课程接口在 episode reset 时调用，返回任务参数更新；当前更新通道支持命令范围。
世界模型和策略未来共用环境，但 SuperMimic 的拟合和模型内展开训练尚未实现。

## 检查框架（不训练）

```bash
uv run --extra train rl-go2 validate
uv run --extra train python -m unittest discover -s tests -v
# 包含 Genesis 初始位姿、局部重置和超时重置的集成测试
RL_GO2_TEST_GENESIS=1 uv run --extra train python -m unittest discover -s tests -v
uv run --extra train rl-go2 smoke --backend mujoco --num-envs 1 --device cpu --steps 16 --check-runner
uv run --extra train rl-go2 smoke --backend genesis --num-envs 2 --device cpu --steps 4 --check-runner
```

smoke 使用零动作，或通过 `--check-runner` 初始化随机 PPO 策略并前向采样；不会调用 learn/update。
Genesis 第一次运行需要编译物理内核，耗时可能较长。
Menagerie 模型的零关节角中性姿态会触发 Genesis 的 qpos0 限位提示；框架在步进前显式设置有效的默认关节角和绝对根位姿，集成测试覆盖此行为。

## 后续平地训练

以下命令会开始训练，本轮搭框架时未执行。默认奖励与参数是起点，需要通过正式实验验证。

```bash
uv run --extra train rl-go2 train --device cuda --num-envs 4096 --iterations 1000 --output runs/flat-ppo-001
uv run --extra train tensorboard --logdir runs
```

CPU 开发调试可用 `--device cpu --num-envs 4`。输出目录须为新目录，避免覆盖结果。
训练目录保存完整解析配置、依赖版本、模型和网格哈希、策略约定、RSL-RL checkpoint 和含观测归一化的 `policy.pt`。
配置快照当前使用本机绝对资产路径；移动到其他机器时须迁移资产并调整路径，不属于独立部署包。
课程当前为 `name: none`；训练恢复和自适应课程留待后续实现。

```bash
# 已训练策略的 sim2sim rollout，默认无窗口
uv run --extra train rl-go2 evaluate --run-dir runs/flat-ppo-001 --backend mujoco --steps 1000
# 在训练后端评估同一策略
uv run --extra train rl-go2 evaluate --run-dir runs/flat-ppo-001 --backend genesis --steps 1000
```

评估使用固定种子和固定难度，但当前尚无保存初始状态/命令序列的正式基准。
需要窗口时加 `--viewer`；macOS MuJoCo 被动 Viewer 需使用其 `mjpython` 启动器运行 `-m rl_go2.cli`。

接口参考：[Genesis](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html)、
[RSL-RL](https://github.com/leggedrobotics/rsl_rl)、[MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html)。
