# RL-Go2

强化学习机械狗作业：Genesis 训练，MuJoCo 做 sim2sim 验证。已跑通 CPU 小规模 PPO 训练与跨引擎策略回放，尚未验证稳定行走。

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
  observations.py    仿真和 SDK 共用的策略观测构造
  hardware/          宇树 SDK2 状态适配、只读订阅、离线命令报文
  tasks/             任务选择、生命周期协议，以及平地速度跟踪实现
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

当前平地任务的策略观测为 45 维：机身坐标角速度(3)、投影重力(3)、速度命令(3)、默认角度偏移(12)、关节速度(12)、上一步裁剪动作(12)。其他任务可以定义自己的观测。
动作经裁剪与缩放后得到默认关节角的偏移，逐物理步执行共享 PD 并限制力矩。当前不模拟执行延迟。
MuJoCo 初期只实现 CPU 单环境评估；Genesis 支持批量环境。设备选择 cpu/cuda/mps，实际训练设备与驱动需在训练前验证。

`Transition` 接口记录重置前的状态转移，单独保留 terminated/truncated。
其中 `context` 是任务提供的目标/阶段张量字典，不要求每个任务都有速度命令。
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

以下命令会开始训练。默认奖励与参数是起点，需要通过正式实验验证。

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

## 添加其他任务

实验配置的 `task` 指向任务 YAML，任务 YAML 的 `name` 在 `src/rl_go2/tasks/__init__.py` 的 `TASK_TYPES` 中选择实现。
当前只注册了 `flat_walk`，跳跃尚未实现。新增任务时：

1. 在 `tasks/` 添加任务类，实现 `Task` 协议：参数校验、任务重置、观测、奖励、终止、步末更新和任务上下文。
2. 在 `TASK_TYPES` 增加名称到 `模块:类名` 的映射，并新增任务 YAML 与实验 YAML。
3. 在任务类的 `observation_terms` 中声明观测项及维数；策略导出使用此任务的定义。

`QuadrupedEnv` 保留物理控制和 episode 计数；任务自行管理命令采样或阶段状态。`reset` 在物理重置后调用，接收重置状态与课程参数更新。
`on_step_end` 在奖励、终止判断和自动重置后调用，用于准备下一步观测；刚重置环境的 episode 计数为零。
走路任务的命令重采样逻辑位于 `tasks/flat_walk.py`，环境不再读取 `commands` 或重采样周期。
任务特有参数由任务类校验，通用配置加载器只检查任务名称及 episode 时长等共同字段。
测试中使用无速度命令、只有 2 维观测的任务，验证同一环境、PPO 适配层及策略元数据支持其他任务定义。

## 已完成的本机流程验证

2026-10-30 在 Apple Silicon Mac 的 CPU 上执行了 4 个 Genesis 环境、5 轮 PPO 更新，共采集 480 条状态转移；日志 loss 均为有限值，网络参数确实发生更新，并成功导出策略。
同一策略在 MuJoCo 上完成 1000 个控制步（20 秒仿真），观测和奖励均为有限值，并执行一次自动重置。
这是运行链路验证，不是稳定行走或迁移性能结论。当前 12 项测试全部通过，包含双仿真后端及 SDK 离线测试。

本机结果保存在 Git 忽略的 `runs/flat-ppo-local-check-001/`：`verification.json` 是验证报告，`train.log` 是训练日志，`policy.pt` 是导出策略，`mujoco-rollout.mp4` 是 6 秒回放。

```bash
OMP_NUM_THREADS=2 uv run --extra train --extra unitree rl-go2 train --device cpu --num-envs 4 --iterations 5 --output runs/flat-ppo-local-check-002
uv run --extra train --extra unitree rl-go2 evaluate --run-dir runs/flat-ppo-local-check-002 --backend mujoco --steps 1000
```

## 宇树 Go2 SDK2

硬件接入独立于仿真后端，采用官方 [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)，
锁定提交 `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5`，使用可选 `unitree` extra。
仿真训练不需要安装这个 extra。SDK 原始依赖要求 `cyclonedds==0.10.2`；Python 3.12 没有对应预编译 wheel，需要原生库与本地构建。

本机已在 `.deps/cyclonedds` 安装原生库；换机器时如无可用 CycloneDDS，可按下面步骤准备（需要 CMake、C 编译器）：

```bash
python3 scripts/bootstrap_cyclonedds.py
export CYCLONEDDS_HOME="$PWD/.deps/cyclonedds"
uv sync --extra train --extra unitree --locked
uv run --extra train --extra unitree python scripts/install_unitree_crc.py
uv run --extra train --extra unitree python scripts/check_unitree.py
```

本地构建的依赖不纳入 Git，源码下载有固定版本及 SHA-256 校验。`install_unitree_crc.py` 补齐官方 SDK 源码中存在、但上游 wheel 未包含的 Linux CRC 动态库及许可证，不修改 SDK Python 代码。
以后运行 SDK 相关命令保留 `--extra unitree`，否则 uv 会移除该可选依赖。

`check_unitree.py` 默认完全离线：构造官方 LowState/LowCmd，验证 45 维观测、CRC 与序列化，不初始化 DDS。
`UnitreeStateReader` 只订阅 `rt/lowstate`，连接参数位于 `configs/hardware/go2.yaml`。
只有显式传入网卡才会连接，例如 `python scripts/check_unitree.py --interface enp2s0`（本轮未执行）。
它会校验消息头和 CRC，并在接收超时时报错；不会创建命令 Publisher，也不会切换机器人运动模式。

SDK 电机槽位为 FR、FL、RR、RL，每腿 hip/thigh/calf；通过名称映射到机器人配置中的标准顺序。
IMU 角速度已经在机身坐标系中，不会再次逆旋转。底层状态只提供 IMU 和关节等字段，不伪造仿真的机身位置或线速度；当前 actor 恰好不依赖这两项。
`Go2Telemetry.observation()` 和仿真任务调用同一个观测函数。

`position_command()` 构造机器人内置 PD 位置控制报文；`torque_command()` 则关闭内置 PD，裁剪力矩后生成外部控制报文，可用于后续对接共享 PD 控制器。两者均只返回 SDK 消息，不发送。
真机策略部署循环、控制权交接、时序/状态新鲜度监测、停机处理和硬件验证尚未实现；这里完成的是 SDK 接口接入，不是可直接上机的部署程序。

```bash
# 离线测试 SDK 与现有框架；加入 RL_GO2_TEST_GENESIS=1 可同时检查 Genesis
uv run --extra train --extra unitree python -m unittest discover -s tests -v
```
