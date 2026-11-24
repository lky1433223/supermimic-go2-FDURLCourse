# RL-Go2

强化学习机械狗作业：Genesis 训练，MuJoCo 做 sim2sim 验证。PPO 已有平地验证；SuperMimic 式引导训练及从零 bootstrap 配方已通过前进/转向跟踪验收，步态质量和复杂地形仍待验证。

## 输出目录约定

今后每次训练使用独立的 `output/<实验目录>/`，该目录整体被 Git 忽略。
训练使用 `scripts/train.py`。省略 `--output` 时，目录名由配置名、训练预算、Git 短提交号和北京时间组成。

```text
output/<实验目录>/
  launch.sh
  launch-config.json
  commit.txt
  train.log
  exit-code
  training/
    config.json
    policy_spec.json
    policy.pt
    model_*.pt
  evaluation/iteration-<轮次>/<task>-<后端>-<机器>-<时间戳>/
    eval-config.json
    summary.json
    nominal/
      assessment.json
      replay.mp4
```

评测统一使用 `scripts/evaluate.py`。用 `--list-tasks` 查看可用 task，用 `--help` 查看所有参数。
用 `--task` 选择评测。`evaluation/tasks/` 下的 task 类声明 `conditions`、`motions`、
`metrics`、`report`、`episode_seconds` 和结束规则。摩擦、质量、质心与动作都直接在对应文件中修改。
训练目录的 `training/config.json` 提供机器人和控制参数。套件默认使用 `configs/scenes/flat.yaml` 的地面和出生位姿，
并沿用该次训练的机器人资产和仿真参数。声明了场景组合的套件使用所选场景，物理工况由评测 Python 定义覆盖。
`conditions.py` 和 `motion_tasks.py` 提供通用数据类。评测套件通过基类的 `build_plan()` 展开工况与动作组合。
生效定义写入 `eval-config.json` 的 `physical_conditions` 和 `motion_tasks`。
`executor.py` 负责唯一的步进循环，`metrics.py` 采集指标。`rollout` 在计划中声明重置后继续，
其他评测在当前试次结束后切换下一项。展开后的试次保存为 `eval-config.json` 中的 `trials`。
权重轮次无法推断时，用 `--iteration` 指定。各 task 都可用 `--viewer` 实时查看，或在 MuJoCo 下加 `--video` 保存同一次仿真的回放。
ICC 无窗口录制使用 `MUJOCO_GL=egl`。显式 `--output` 指向本次评测的新目录。
录制时可用 `--video --video-view front` 选正面视角，`rear`、`left`、`right` 分别为后、左、右。
这四种预设随机器狗的朝向转动。`default` 沿用固定朝向的斜侧跟拍。
用 `--video --video-view front rear left right` 可将同一次仿真的四个视角录进一个 MP4。
视角按参数顺序从左到右、从上到下排列。单选为整屏，双选为左右并排，三到四选为两行两列，五选为两行三列。
单视角为 640×480，多视角每格为 448×336。录像统一使用 20 FPS、H.264、CRF 24 和 2 Mbit/s 码率上限。
`eval-config.json` 的 `video_views` 和 `video_encoding` 分别记录实际视角顺序和编码设置。

### 标准速度跟踪评测

```bash
uv run --extra train python scripts/evaluate.py --run-dir output/flat-supermimic-guided-001 --task standard
```

每组物理参数跑 27 项，横向速度指令固定为 0。速度单位为 m/s，转向单位为 rad/s。
标称组使用零质心偏移、标称质量与惯量和四脚滑动摩擦 0.8。另有两个以固定种子抽出的物理样本，
质心 X/Y/Z 偏移范围为 ±1/±3/±1 cm，质量倍率为 0.9–1.1，四脚共用摩擦倍率为 0.75–1.25。
这些样本在所有动作中保持固定，每个评测种子共跑 81 项。

| 组别 | 前进速度 | 转向速度 | 项数 |
| --- | --- | --- | --- |
| 常规 | 0.2、0.4、0.6、0.8 | 0、±0.4 | 12 |
| 常规 | 0.5 | 0、±0.1、±0.3、±0.5 | 7 |
| OOD | 0 | 0 | 1 |
| OOD | 0.5 跑 8 秒，再切到 0 | 0 | 1 |
| OOD | 1.0、1.2 | 0 | 2 |
| OOD | 0.5 | ±0.6、±0.8 | 4 |

每项从默认姿态开始，最多运行 20 秒。摔倒时记录时间并结束该项，继续后续工况。
报告包含全程及去掉前 2 秒后的速度 RMSE，以及走停工况的稳定用时和末 2 秒位移。
`standard` 默认用种子 1，可用 `--seeds` 对整套工况复测。

### 固定低摩擦评测

`low-friction` 使用五组固定滑动摩擦：四脚统一 0.8、0.1、0.2、左侧低摩擦和右侧低摩擦。
每组跑静站、0.2 m/s 直行、0.5 m/s 直行、以 0.5 m/s 左右各转一次（偏航 ±0.4 rad/s）和走停。
每项最多 20 秒，走停在第 8 秒清零指令。五组摩擦分别配标称机身和与 `standard` 共享的两个质心和质量样本，
每个评测种子共 90 项。

```bash
uv run --extra train python scripts/evaluate.py --run-dir output/flat-supermimic-guided-001 --task low-friction
```

### 行进中切换摩擦

摩擦切换对照使用 `python scripts/evaluate.py --task friction-switch --run-dir <训练目录>`。
它与 `low-friction` 共用六项动作和三个机身样本。四条滑动摩擦曲线分别是固定 0.8、
0.8 → 0.4 → 0.8、0.8 → 0.2 → 0.8，以及中间 0.2 只持续 2 秒的短时低摩擦。
低摩擦都从第 5 秒开始，长段在第 10 秒恢复，短段在第 7 秒恢复。每个评测种子共 72 项。
报告记录逐项跟踪、走停指标以及各阶段的实际接触摩擦和跟踪误差。
训练时可在启用 `foot_ground_friction` 后设置 `env_multiplier_range: [0.75, 1.25]` 和
`switch_interval_seconds: [2.0, 5.0]`。启用定时切换时四脚基础范围须为固定值，
各环境独立重抽倍率，真实采样和想象 rollout 使用相同的倍率与时长分布。

`rollout` 用于运行检查，固定零质心偏移、标称质量和惯量、四脚滑动摩擦 0.8 和 0.5 m/s 直行。
默认用种子 1 跑 1000 个控制步，重置后继续至步数上限。

### 机身质量和质心泛化

`--task mass` 固定零质心偏移，依次测试机身质量倍率 0.50、0.80、0.90、1.00、1.10、1.20、1.50。
惯量随质量同比缩放。每个种子跑 42 项。

`--task com-offset` 固定标称质量与惯量，测试零偏移、X 轴 ±1/±3/±5 cm、Y 轴 ±3/±4 cm、
Z 轴 ±1 cm，以及三轴同时偏移 (+3, +3, +1) cm。每个种子跑 84 项。

两项都使用平地步态、固定四脚滑动摩擦 0.8 和与 `low-friction` 相同的六项动作。
每项最多 20 秒。报告按工况统计摔倒，并记录实际机身质量、质心偏移、速度跟踪和走停指标。
静站另记录机身最低高度和最终水平位移。

### 随机点力恢复

`--task point-push` 在 MuJoCo 标称物理工况下运行相同的六个动作。对照组无推力，机身组为 20 N、30 N、80 N 和 160 N，
足端组为 8 N 和 12 N，均持续 0.2 秒。机身 80 N 和 160 N 各增加一组持续 0.1 秒的短推力对照。
推力在第 5 秒开始，走停项在第 10 秒开始。默认种子 1、2、3，共 162 项。
同一受力部位的不同力度和时长复用对应试次的施力连杆、局部落点和世界系方向。
`eval-config.json` 保存实际点力事件，报告按工况统计摔倒及恢复结果。恢复时间从撤力到满足 0.5 秒稳定判据为止。

用 `--condition` 按工况名筛选，用 `--trial` 按动作名进一步筛选。评测报告和可选录像只包含选中的试次。
下面的命令只用种子 1 录制 80 N 短推下的慢速直行。去掉 `--trial` 和 `--seeds` 就会覆盖所选工况下的全部动作和默认种子。

```bash
python scripts/evaluate.py --task point-push --run-dir <训练目录> \
  --condition base-80n-short --trial vx0.2_yaw0 --seeds 1 --video
```

### 地形场景对照

`python scripts/evaluate.py --task terrain --run-dir <训练目录>` 用同一策略和标称物理工况在平地、2/4/6 cm 台阶、5°/10° 缓坡和两张随机不平整地面上执行 8 秒直行。
可加 `--backend genesis` 用同一套件检查 Genesis。`eval-config.json` 的 `scenes` 记录解析后的碰撞几何和机器人出生位姿，
`summary.json` 按场景汇总通过、失败及摔倒次数。各场景的报告和可选录像保存在 `<场景>/<工况>/`。
通过要求全程未摔倒，且去掉前 2 秒后的前进速度 MSE 不超过 0.04 (m/s)²（RMSE 0.20 m/s）。
逐项报告记录失败原因、最大倾角和前进位移。

`configs/scenes/flat.yaml` 使用单个平面，`configs/scenes/single_step.yaml` 在平面上增加一个固定箱体。
台阶的 `height_m` 按米配置。调整高度时，箱体前沿和底面位置固定。
`configs/scenes/ramp.yaml` 包含上坡、平台和下坡。`slope_deg` 控制坡度，`run_m` 是每段坡的水平长度，平台高度由这两个参数决定。
训练配置可用 `scene: ../scenes/single_step.yaml` 或 `scene: ../scenes/ramp.yaml` 选择地形。
评测套件在 `EvaluationGroup` 中选择场景、工况与动作，可通过 `SceneSpec.entity_overrides` 复用地形文件并指定多个高度或坡度。
当前地形套件使用平地步行任务的奖励和终止规则。

`configs/scenes/rough_ground.yaml` 用 `seed` 生成一张 5×4 的高度图。每块砖默认边长 30 cm，高度为 0、1 或 2 cm，
相邻高差与入口、出口到平地的高差最多为 1 cm。`terrain` 使用地形种子 101 和 102，地形种子独立于 `--seeds`。
`eval-config.json` 和对应场景的 `assessment.json` 保存地形种子和实际高度图。
训练配置可用 `scene: ../scenes/rough_ground.yaml` 加载该文件指定的地图，地图在环境创建时生成。

训练启动时会打印绝对输出路径。历史 `runs/` 及当前已有产物保持原位，仍被忽略，不自动迁移。

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
  curriculum.py      固定难度和按采样步数推进的推力课程
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
课程接口在 episode reset 时调用，返回任务参数更新；当前任务支持命令范围或推力范围更新。
SuperMimic 已接入同一环境，使用 replay 多步拟合和模型内可微策略更新；见下文。

## 检查框架（不训练）

```bash
uv run --extra train python scripts/check_env.py validate
uv run --extra train python -m unittest discover -s tests -v
# 包含 Genesis 初始位姿、局部重置和超时重置的集成测试
RL_GO2_TEST_GENESIS=1 uv run --extra train python -m unittest discover -s tests -v
uv run --extra train python scripts/check_env.py smoke --backend mujoco --num-envs 1 --device cpu --steps 16 --check-runner
uv run --extra train python scripts/check_env.py smoke --backend genesis --num-envs 2 --device cpu --steps 4 --check-runner
```

smoke 使用零动作，或通过 `--check-runner` 初始化随机 PPO 策略并前向采样；不会调用 learn/update。
Genesis 第一次运行需要编译物理内核，耗时可能较长。
Menagerie 模型的零关节角中性姿态会触发 Genesis 的 qpos0 限位提示；框架在步进前显式设置有效的默认关节角和绝对根位姿，集成测试覆盖此行为。

## 后续平地训练

以下命令会开始训练。默认奖励与参数是起点，需要通过正式实验验证。

```bash
uv run --extra train python scripts/train.py --device cuda --num-envs 4096 --iterations 1000 --output output/flat-ppo-001
uv run --extra train tensorboard --logdir output
```

CPU 开发调试可用 `--device cpu --num-envs 4`。输出目录须为新目录，避免覆盖结果。
训练目录保存完整解析配置、依赖版本、模型和网格哈希、策略约定、RSL-RL checkpoint 和含观测归一化的 `policy.pt`。
配置快照当前使用本机绝对资产路径；移动到其他机器时须迁移资产并调整路径，不属于独立部署包。
平地任务的课程为 `name: none`；支持用 `--init-checkpoint` 初始化权重，但不恢复优化器、迭代计数或课程进度。

```bash
# 已训练策略的 sim2sim rollout，默认无窗口
uv run --extra train python scripts/evaluate.py --task rollout --run-dir output/flat-ppo-001 --backend mujoco --steps 1000
# 在训练后端评估同一策略
uv run --extra train python scripts/evaluate.py --task rollout --run-dir output/flat-ppo-001 --backend genesis --steps 1000
```

评估使用固定种子和固定难度，但当前尚无保存初始状态/命令序列的正式基准。
需要窗口时加 `--viewer`。macOS 的 MuJoCo 窗口用 `mjpython` 启动 `scripts/evaluate.py`。
如果 uv Python 的 `mjpython` 找不到 `libpython3.12.dylib`，将基础 Python 的 `lib/` 加入 `DYLD_FALLBACK_LIBRARY_PATH`。

接口参考：[Genesis](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html)、
[RSL-RL](https://github.com/leggedrobotics/rsl_rl)、[MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html)。

## 添加其他任务

实验配置的 `task` 指向任务 YAML，任务 YAML 的 `name` 在 `src/rl_go2/tasks/__init__.py` 的 `TASK_TYPES` 中选择实现。
当前注册了 `flat_walk` 和 `push_walk`，跳跃尚未实现。新增任务时：

1. 在 `tasks/` 添加任务类，实现 `Task` 协议：参数校验、任务重置、观测、奖励、终止、外力、步末更新和任务上下文。没有扰动时 `external_force` 返回 `None`。
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
OMP_NUM_THREADS=2 uv run --extra train --extra unitree python scripts/train.py --device cpu --num-envs 4 --iterations 5 --output output/flat-ppo-local-check-002
uv run --extra train --extra unitree python scripts/evaluate.py --task rollout --run-dir output/flat-ppo-local-check-002 --backend mujoco --steps 1000
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

## 抗推扰任务及课程

`configs/experiments/push_ppo.yaml` 选择 `push_walk`。它保持平地任务的 45 维观测、控制与奖励，训练时每隔 3–6 秒，从世界坐标系的 ±X/±Y 方向随机施加 0.2 秒外力。力作用于基座连杆质心，单位 N，每个物理子步重新施加，停止后清零，不通过改速度伪造推力。
任务为下一控制步生成推力；推力不进入 actor 观测，但会记录到模型学习用的 `Transition.context.external_force_world`。

按新收集的状态转移数量推进课程，新强度范围只在 episode reset 时生效：

| 状态转移数量 | 推力范围 |
|---|---|
| 0–1000 万 | 5–20 N |
| 1000–3000 万 | 20–40 N |
| 3000–6000 万 | 40–60 N |
| 6000 万以后 | 60–80 N |

用完成的平地 checkpoint 初始化 actor、critic 及其归一化状态，重新初始化优化器、迭代和课程。会检查关节顺序、控制参数、时间步和观测布局是否一致；学习率从 0.0003 开始。关闭初始 episode 计数随机化，避免初次落地时触发本应延后的推力。

```bash
uv run --extra train python scripts/train.py --config configs/experiments/push_ppo.yaml --device cuda --num-envs 4096 --iterations 1000 --init-checkpoint output/flat-ppo-icc-gpu2-001/training/model_999.pt --output output/push-ppo-001
```

受推基线使用已完成的 1000 轮平地策略。在 MuJoCo 中固定 0.3/0.5/0.7 m/s 命令，各次测试在第 3 秒施力 0.2 秒；每个非零强度分别测试世界坐标四个方向，共 12 个条件。每次最多 8 秒，摔倒立即结束。
恢复定义为水平速度误差小于 0.1 m/s、偏航角速度绝对值小于 0.15 rad/s、倾角小于 15°，并连续保持 0.5 秒。恢复耗时从推力结束开始计算；均值只统计成功恢复的条件，未恢复/摔倒条件单独计数。

| 推力 | 摔倒次数 / 固定测试条件数 |
|---|---|
| 0 N | 0 / 3 |
| 20 N | 1 / 12 |
| 40 N | 6 / 12 |
| 60 N | 8 / 12 |
| 80 N | 9 / 12 |
| 100 N | 9 / 12 |

这是有限固定条件的基线，不是总体成功率估计。报告和视频保存在本机忽略目录 `runs/push-baseline-001/`。

同一基线已在 ICC CPU 复测，各强度下摔倒计数一致，本机副本为 `runs/push-baseline-001/icc-assessment.json`。
已在 ICC 的 GPU 2 启动 4096 环境、1000 轮抗扰动训练，结果目录 `/home/lky/RL-Go2/runs/push-ppo-icc-gpu2-001`，tmux 会话为 `rl-go2-push-gpu2`。
日志和启动参数位于 `/home/lky/RL-Go2/runs/launches/icc-push-ppo-gpu2-001/`，该目录保留带 SHA-256 的源码快照。
启动脚本在训练成功、策略导出后，自动在 CPU 上运行同一受推评测，写入结果目录的 `push-assessment/assessment.json`；不使用 GPU 3–6。

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
从零 GPU 配方的历史版本也通过了下面记录的速度跟踪验收；
它是本项目的 SuperMimic 式实现，尚无证据优于 PPO，也不能据速度指标认定步态已经正常。

算法参考 [SuperMimic](https://arxiv.org/abs/2610.28480) 和
[Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic)，相关 MIT 许可见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

### 复现与独立验收

```bash
uv sync --extra train --locked
# 需与当前 Go2 的关节顺序、PD、动作缩放和 45 维观测兼容的 TorchScript 策略。
# 本机已验证 driver 的绝对路径见下方说明；仓库不附带教师权重。
uv run --extra train python scripts/train.py --config configs/experiments/flat_supermimic_guided.yaml \
  --driver-policy /absolute/path/to/compatible-go2-policy.pt --output output/flat-supermimic-guided-001
uv run --extra train python scripts/evaluate.py --run-dir output/flat-supermimic-guided-001 --task standard
uv run --extra train python scripts/evaluate.py --run-dir output/flat-supermimic-guided-001 --task standard \
  --seeds 201 202 203 204 205 206 207 208 209 210
# 导出策略推理不需要 --driver-policy，也不会读取教师模型。
uv run --extra train python scripts/evaluate.py --task rollout --run-dir output/flat-supermimic-guided-001 --backend mujoco --steps 1000
```

已验证配置为 Genesis CPU、32 环境、900 轮：16 个环境由当前 actor 采集，16 个由 driver 采集。
前 600 轮每轮拟合世界模型、执行 4 次监督 actor 更新；随后 300 轮优化 32 步 imagined reward，并用真实状态上的行为损失约束偏移。
世界模型隐藏层为 128×128，actor 为 128×128，训练窗口为 8 步。
预热 actor 学习率 0.001，模型阶段 actor 学习率 0.00002，行为约束权重 5.0。
`iterations: 900` 是固定验证协议确定的早停配置；更长训练会出现 sim2sim 退化，不能仅按 imagined return 选择 checkpoint。
GPU 可覆盖 `--device cuda --num-envs ...`，但不同规模尚未重新调优，不能直接继承本机性能结论。

`flat_supermimic.yaml` 保留无监督预热的研究对照，默认不使用 driver，不是已证明能学会步态的推荐配方。
两种配置默认 `bootstrap_weight: 0`，采用短期奖励和 soft survival 的消融目标。
设为 1 可启用末端 critic 和 imagined TD(lambda) 拟合；新增 GPU 配方使用该设置。
`check_supermimic.py` 显式启用 critic，30 轮只验证更新、导出和仿真链路，不用于速度性能验收。

### 从零 GPU 配方：关节限位与 critic bootstrap

`flat_supermimic_bootstrap.yaml` 使用 4096 环境、40000 个采样轮，不加载教师或旧权重。
当前配方从第 1 轮起计入 replay 采样；有足够帧后按官方调度每 5 个采样轮约更新 2 次。
每次更新用 10 批 × 2048 个同回合 8 帧窗口拟合世界模型，每个窗口监督后 7 帧，
随后用 4096 条 32 步预测轨迹训练 actor 和 critic。Critic 的 5 个 epoch 复用同一次打乱。
关节目标角按 `go2.yaml` 的物理范围向内留出 0.05 rad，
真实仿真、replay 和 imagined rollout 均使用裁剪后的动作；历史配置没有限位字段时保留原行为。

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra train python scripts/train.py \
  --config configs/experiments/flat_supermimic_bootstrap.yaml --output output/flat-supermimic-bootstrap-001
uv run --extra train python scripts/evaluate.py --run-dir output/flat-supermimic-bootstrap-001 --task standard
```

2026-11-02 ICC GPU 1 的一次完整训练使用旧调度（300 轮世界模型预热、每轮 2 个模型 batch、
8192 条预测轨迹）。该策略在 MuJoCo 5 个种子共 300 秒测试中零摔倒，
vx/vy/yaw RMSE 分别为 0.0586 m/s、0.0179 m/s、0.0404 rad/s。
该结果来自单次训练，不包含步态质量验收；模型、日志和视频均保留在被 Git 忽略的 `output/` 内。

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
历史验收标准为零摔倒，且三个聚合 RMSE 各小于 0.15（对应 SI 单位）。逐种子数据见历史评测结果。

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
旧调度的窗口保留终止/超时当步的模型目标，并屏蔽 reset 后的后续步骤。
无独立预热的配方按官方固定倍率 16 调度，抽取完整同回合窗口，以第一帧为起点监督后续帧。

支持模型训练的任务额外提供以下纯批量接口：

- `observe(state, previous_action, context)`、`reward(..., context)`：显式读取采样上下文，复用真实任务公式。
- `model_features(state, context) -> [batch, F]`：任务附加特征，例如随机化的物理参数，宽度在初始化时推导。
- `advance_context(context, next_state)`：推进命令/阶段，可根据预测世界位姿准备扫描；不得修改真实任务状态。
- `termination_probability(...)`：模型内可微终止概率，真实环境仍使用硬阈值。
- 可选 `metrics(state, context)`：每个值首维为 batch 的物理指标字典；训练日志只聚合 actor 环境，driver 回报单独记录。

实验配置选择含箱体的地形或多地形集合时，世界模型从解析后的碰撞几何查询局部地面。
扫描以机身航向为坐标系，前后偏移为 −0.30 到 +0.50 m，左右偏移为 −0.25 到 +0.25 m，间隔均为 0.10 m。
54 个值按前后行、左右列排列，每个值为地面 Z 减去机身 Z。格内读取实际砖高，地块外读取平面高度。
归一化统计按真实位姿查询，多步模型拟合和想象 rollout 在后续步按预测位姿重新查询。Replay 保留世界位姿，地图写入生效配置。
Actor、Critic、奖励和终止仍使用原有定义。平地 checkpoint 可通过 `--init-checkpoint` 初始化，新高程输入的首层权重从零开始。
54 点扫描读取场景中固定箱体的实际表面，支持台阶、缓坡和随机砖块。

`configs/experiments/terrain_supermimic_push.yaml` 提供多地形联合泛化配方。
`scene.terrains` 以 `flat`、`step`、`ramp`、`rough_ground` 为键，声明本次训练要生成的地形。
`weight` 决定每回合选择各类别的相对概率，类别内部均匀选择地块。
各类可通过 `map_count` 指定地块数量，默认平地 1 块，其余各 8 块。
台阶的 `height_range_m`、缓坡的 `slope_range_deg` 在启动时采样具体参数，随机地面使用连续种子生成。
生成结果与各地块出生位姿写入生效配置，地块沿 Y 方向间隔 20 m。
环境重置时按权重重新选择地块并设置出生位姿。单次运行的碰撞几何固定。
实验中的物理随机化和速度任务继续由 `domain_randomization` 与 `task` 配置。
地形课程和基于地形的奖励、终止规则留待后续设计。
若引入执行延迟，需把真正的执行器状态加入动力学状态，不能用无约束 previous_action 快捷输入代替。

输出目录必须不存在，保存完整配置、策略约定、资产哈希、逐轮 `metrics.jsonl` 和网络/优化器 checkpoint。
世界模型修正后的 checkpoint schema 为 `rl-go2.supermimic/v2`，与旧输入维度不兼容。
当前没有恢复训练 CLI，也不保存完整 replay；本节的“引导预热”是监督训练，不是加载另一 actor 的权重。

### 交互式策略推理

`play.py` 持续运行单个机器人的策略推理，默认 1 倍真实时间。
聚焦仿真窗口即可接收键盘输入，窗口内 HUD 显示速度和状态，失焦清空移动按键。
MuJoCo 使用官方渲染 API 和 GLFW 窗口接收完整的按下/松开事件，Genesis 使用原生 Viewer 插件。

```bash
# play.py 的原生窗口在主线程创建，两个后端均使用普通 python。
uv run --extra train python scripts/play.py --backend mujoco \
  --run-dir output/<训练目录> --iteration <权重轮次>
uv run --extra train python scripts/play.py --backend genesis \
  --run-dir output/<训练目录> --iteration <权重轮次>
```

W/S 按住前进/后退，A/D 按住左转/右转，支持组合键。↑/↓ 每次增减
0.1 m/s 步速，←/→ 每次减增 0.1 rad/s 转速。Space 清空移动指令，
P 暂停/继续，R 重置机器人，Esc 退出。指令以 0.15 秒时间常数平滑变化，
停止指令时策略继续执行。窗口内 HUD 并排显示目标和机体坐标系实际速度，并单列速度档位和鼠标施力。
按 F1 展开策略路径、训练指令范围、仿真时间和物理条件。
按 N 循环切换地形，HUD 显示当前地块和进度。含多地形的训练使用已保存的地块，旧版训练可使用平地、台阶、斜坡和粗糙地面预设。
切换时重置机器人并移动视角。手动重置和跌倒恢复都使用当前地块。
HUD 使用英文，鼠标左键拖动旋转视角、右键拖动平移、滚轮缩放。
Genesis 额外启用鼠标弹簧施力：左键点中机器人部件并拖动可牵拉该部件，
松开释放；拖动空白处调整视角。失焦、暂停、重置或停走会释放抓取。
线速度和角速度档位上限采用对应训练范围端点的最大绝对值，允许正反向操作，
训练未覆盖的方向（例如仅前进训练后的后退）效果需要实际观察。

默认读取 `training/policy.pt` 和 `training/config.json`，也兼容旧版训练目录。
可用 `--checkpoint` 加载 SuperMimic checkpoint，或 `--policy` 指定导出策略。
跨机器的失效资源路径会按仓库 `assets/` 路径重新定位，源配置保持原样。
物理条件固定为标称质量、零质心偏移和四脚 0.8 滑动摩擦。跌倒后自动重置并清空指令。
`--steps` 可限制交互短跑步数。

MuJoCo 也支持左键点中机器人部件并拖动施力，HUD 显示抓取部件和鼠标施力大小（N）。
松开鼠标、窗口失焦、暂停、重置或自动跌倒恢复时释放抓取，空白处仍可拖动视角。
