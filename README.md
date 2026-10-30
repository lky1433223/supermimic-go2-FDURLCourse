# RL-Go2

强化学习机械狗作业：Genesis 训练，MuJoCo 做 sim2sim 验证。PPO 已有平地验证；SuperMimic 式引导训练已通过前进/转向跟踪验收，纯从零模型训练和复杂地形仍未验证成功。

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
  trainers/supermimic.py  世界模型、模型内 actor–critic 训练与策略导出
  trainers/model_learning.py  按环境 replay、局部坐标动力学和归一化
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
SuperMimic 已接入同一环境，使用 replay 多步拟合和模型内可微 actor–critic 更新；见下文。

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

## ICC 训练及第 700 轮 sim2sim 评测

2026-10-30 将提交 `62092f0` 部署到 ICC 的 `/home/lky/RL-Go2`，用 uv 安装锁定的训练依赖。
训练配置为 Genesis、4096 个并行环境、1000 轮 PPO；使用物理 GPU 2（A100 80GB），通过 `CUDA_VISIBLE_DEVICES` 的 UUID 限制可见设备，未使用 GPU 3–6。
服务器 Python 为 3.12.3，PyTorch 为 2.13.0+cu130；通过 CUDA 张量计算和 32 环境仿真检查后，在 tmux 会话 `rl-go2-flat-gpu2` 中启动训练。
服务器结果目录为 `/home/lky/RL-Go2/runs/flat-ppo-icc-gpu2-001`，日志为 `/home/lky/RL-Go2/runs/launches/icc-flat-ppo-gpu2-001/train.log`。

取第 700 轮 checkpoint，在本机 MuJoCo 中使用相同机器人资产、控制和任务配置，评估随机命令种子 1–5，每组 20 秒，共 5000 个控制步、100 秒仿真：

| 指标 | 结果 |
|---|---:|
| 摔倒终止次数 | 0 |
| 前进速度 RMSE | 0.06846 m/s |
| 横向速度 RMSE | 0.02720 m/s |
| 转向角速度 RMSE | 0.04504 rad/s |
| 平均回报 | 28.50337 |

评测时保持 checkpoint 权重不变，没有在 MuJoCo 微调。结果说明该 checkpoint 在这五组平地命令下能够持续运动并跟踪速度；尚未验证复杂地形、教室场景或真机表现。
本机详细报告、评测脚本、模型和 12 秒视频位于 Git 忽略的 `runs/icc-flat-ppo-checkpoint-700/`，本提交只记录参数和结果，不包含训练产物。

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

## SuperMimic Go2 平地速度跟踪

已跑通的方案是 **PPO 引导预热 + 行为约束下的可微世界模型优化**。actor 随机初始化，不复制 PPO 网络权重；
训练时用已有 PPO 策略采集一部分走路数据并提供监督动作，部署时只运行独立 actor。
纯从零的模型训练对照仍未达到跟踪验收，不能把这里的结果称为无专家先验的原论文复现，也没有证据优于 PPO 或单独监督预热。

算法参考 [SuperMimic](https://arxiv.org/abs/2610.28480) 和
[Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic)，相关 MIT 许可见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

### 复现与独立验收

```bash
uv sync --extra train --locked
# 需与当前 Go2 的关节顺序、PD、动作缩放和 45 维观测兼容的 TorchScript 策略。
# 本机已验证 driver 的绝对路径见下方说明；仓库不附带教师权重。
uv run --extra train rl-go2 train --config configs/experiments/flat_supermimic_guided.yaml \
  --driver-policy /absolute/path/to/compatible-go2-policy.pt --output runs/flat-supermimic-guided-001
uv run --extra train python scripts/evaluate_tracking.py --run-dir runs/flat-supermimic-guided-001 \
  --output runs/flat-supermimic-guided-001/tracking-validation.json --require-tracking
uv run --extra train python scripts/evaluate_tracking.py --run-dir runs/flat-supermimic-guided-001 \
  --seeds 201 202 203 204 205 206 207 208 209 210 \
  --output runs/flat-supermimic-guided-001/tracking-heldout.json --require-tracking
# 导出策略推理不需要 --driver-policy，也不会读取教师模型。
uv run --extra train rl-go2 evaluate --run-dir runs/flat-supermimic-guided-001 --backend mujoco --steps 1000
```

已验证配置为 Genesis CPU、32 环境、900 轮：16 个环境由当前 actor 采集，16 个由 driver 采集。
前 600 轮每轮拟合世界模型、执行 4 次监督 actor 更新；随后 300 轮优化 32 步 imagined reward，并用真实状态上的行为损失约束偏移。
世界模型隐藏层为 128×128，actor 为 128×128，训练窗口为 8 步。
预热 actor 学习率 0.001，模型阶段 actor 学习率 0.00002，行为约束权重 5.0。
`iterations: 900` 是固定验证协议确定的早停配置；更长训练会出现 sim2sim 退化，不能仅按 imagined return 选择 checkpoint。
GPU 可覆盖 `--device cuda --num-envs ...`，但不同规模尚未重新调优，不能直接继承本机性能结论。

`flat_supermimic.yaml` 保留无监督预热的研究对照，默认不使用 driver，不是已证明能学会步态的推荐配方。
两种配置默认 `bootstrap_weight: 0`，采用短期奖励和 soft survival 的消融目标。
设为 1 可启用末端 critic 和 imagined TD(lambda) 拟合，但该分支尚未验证稳定跟踪，不能用于声明训练成功。
`check_supermimic.py` 显式启用 critic，30 轮只验证更新、导出和仿真链路，不用于速度性能验收。

### 为什么旧训练没有改善

- **覆盖不足。** 旧检查只有 9632 条转移，多数接近站立。采样轨迹上旧 actor 动作标准差约 0.10，而走路 driver 约 0.96。
  旧模型在走路轨迹上 8 步前进速度预测误差约 0.19 m/s；单步损失下降没有证明步态动力学正确。
- **critic 的梯度主导了错误方向。** 一个 64 状态 probe 中，完整目标梯度范数 4.23，短期奖励梯度 0.055，方向余弦 −0.79。
  小规模模型误差与只在 imagination 中拟合的价值估计会相互放大。平地配方先关闭这个末端价值目标。
- **动力学输入不符合当前控制器的因果性。** 无执行延迟的外部 PD 在完整 RobotState 下不依赖 previous_action，旧模型却学到了显著依赖。
  已移除该输入；previous_action 仍保留在策略观测和动作平滑奖励里。回归测试验证其对模型输出没有影响。
- **模型优化仍会偏离可靠步态。** 仅加入 driver 数据或去掉 critic，未使当前小规模纯模型训练达标。
  引导预热解决冷启动；较弱行为约束在继续更新后仍出现摔倒，因此采用更强约束、较小步长和真实仿真早停。

当前 actor 目标是 `-imagined_return + behavior_weight * MSE(actor(real_obs), driver(real_obs))`。
模型阶段仍会计算并应用穿过冻结世界模型的奖励梯度；性能改善主要来自引导预热，尚未证明模型阶段优于预热基线。

### 实际结果与范围

训练后的 actor 独立运行。MuJoCo 固定种子 1–5，各 20 秒，统计全部控制步，包含起步和命令切换：

| 策略 | 前进 RMSE (m/s) | 横向 RMSE (m/s) | 转向 RMSE (rad/s) | 摔倒 |
|---|---:|---:|---:|---:|
| 旧 600 轮检查 | 0.5502 | 0.0293 | 0.2785 | 0 |
| PPO 引导策略 | 0.0685 | 0.0272 | 0.0450 | 0 |
| 引导预热后、尚无模型策略更新 | 0.0810 | 0.0260 | 0.0450 | 0 |
| 最终 900 轮独立 actor | 0.0819 | 0.0259 | 0.0475 | 0 |

冻结早停选择后，新种子 201–210 共 200 秒独立测试，前进/横向/转向 RMSE 为
0.0783 m/s、0.0248 m/s、0.0512 rad/s，零摔倒。
训练后端 Genesis 单环境种子 1 的 20 秒独立检查同样通过；前进/转向 RMSE 为 0.0558 m/s 和 0.0157 rad/s。
验收标准在比较前固定为零摔倒，以及三个聚合 RMSE 各小于 0.15（对应 SI 单位）；脚本保存逐种子结果，失败时 `--require-tracking` 返回非零。

范围是平地前进命令 0.2–0.8 m/s、转向命令 ±0.5 rad/s，每 4 秒重采样；横向命令固定为 0。
横向 RMSE 衡量漂移，不代表已验证侧向行走。尚未验证复杂地形、外力扰动或真机部署。

本机最终目录是 `runs/supermimic-flat-guided-003/`（Git 忽略）：`policy.pt` 可独立推理，
`model_900.pt` 保存训练状态，`verification.json` 汇总检查，`tracking-validation.json` 和 `tracking-heldout.json` 保存评估，
`mujoco-tracking.mp4` 是带目标/实际速度的 12 秒回放，`video-trace.json` 保存对应数据。
本机 driver 位于 `/Users/shiinarikka/PycharmProjects/RL-Go2/runs/icc-flat-ppo-checkpoint-700/policy.pt`，其 SHA-256 保存在验证报告中。
完整复跑的 actor 与世界模型和冻结选择时的 900 轮 checkpoint 逐张量一致。
23 项测试全部通过，包含 Genesis 集成测试、梯度冻结、因果性、driver 不进入导出策略、预热学习率恢复，以及原 PPO/SDK 回归。

### 开发接口与后续地形

Actor 保持 45 维策略观测和 12 维有界动作，沿用现有 PD、关节顺序和真实奖励。
世界模型看到去除水平位置和 heading 的局部 Markov 状态与当前动作，保留预测世界位姿用于地形查询；
同时监督预测最后一个物理子步的力矩，用于原有力矩奖励，未替换成控制步末 PD 估计。
Replay 保存实际执行的裁剪动作、previous_action、上下文、reward、driver 来源和重置前 next_state。
终止/超时当步的模型目标保留，只屏蔽窗口中 reset 后的后续步骤。

支持模型训练的任务额外提供以下纯批量接口：

- `observe(state, previous_action, context)`、`reward(..., context)`：显式读取采样上下文，复用真实任务公式。
- `model_features(state, context) -> [batch, F]`：物理参数、局部高度扫描等；平地返回空特征，宽度在初始化时推导。
- `advance_context(context, next_state)`：推进命令/阶段，可根据预测世界位姿准备扫描；不得修改真实任务状态。
- `termination_probability(...)`：模型内可微终止概率，真实环境仍使用硬阈值。
- 可选 `metrics(state, context)`：每个值首维为 batch 的物理指标字典；训练日志只聚合 actor 环境，driver 回报单独记录。

Context 为固定键集合的小型张量字典，replay 自动保存；地形图由任务持有，context 仅记录地图索引等数据。
模型内逐步从预测位姿重算物理特征；当前拟合阶段使用 replay 的逐步特征，新增地形时必须验证二者分布的一致性。
新任务仍需实现地形资产、相对地面的高度奖励/终止、接触几何、重置和课程，不能只增加网络输入维数。
若引入执行延迟，需把真正的执行器状态加入动力学状态，不能用无约束 previous_action 快捷输入代替。

输出目录必须不存在，保存完整配置、策略约定、资产哈希、逐轮 `metrics.jsonl` 和网络/优化器 checkpoint。
世界模型修正后的 checkpoint schema 为 `rl-go2.supermimic/v2`，与旧输入维度不兼容。
当前没有恢复训练 CLI，也不保存完整 replay；本节的“引导预热”是监督训练，不是加载另一 actor 的权重。
