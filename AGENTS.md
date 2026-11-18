# 项目输出约定

- 新训练和实验产物统一放在 `output/<每次训练的独立目录>/`，不要散放在 `output/` 根目录。
- 同一次训练的模型、配置、日志、指标、评测报告和回放视频放在该训练目录内；后续评测使用其 `evaluation/` 子目录。
- `output/` 整体由 Git 忽略，不提交模型、日志或视频。
- 历史 `runs/` 和已有产物保留原位，除非用户要求迁移。

## 执行机器与同步

- 使用 `python scripts/train.py` 训练、`python scripts/evaluate.py` 评测、`python scripts/check_env.py` 检查环境。评测协议为 `standard`、`friction-switch` 和 `rollout`。

- 训练、评测和视频录制必须使用仓库已有的脚本或命令入口。禁止用临时脚本、内联 Python、经 SSH 标准输入执行的代码等方式绕过现有入口。
- 如果现有入口无法完成任务，先告诉用户缺少什么功能，并在对话里讨论如何接入仓库。等用户确认方案后再修改代码和运行实验。

- 正式训练、MuJoCo 评测和视频录制默认都在 ICC 执行，主产物保存在 `/home/lky/RL-Go2/output/`。用户明确指定本地执行时，在本地使用相同的相对路径结构。
- 如实际需要换机器执行，先向用户说明哪些步骤需要换机器以及原因。汇报结果时注明训练、评测指标和视频分别在哪台机器生成；本地文件链接只代表文件所在位置。
- 用户要看报告或视频时，将所需文件同步到本地同一相对路径，并提供本地链接。完整 checkpoint 留在 ICC，仅在需要本地推理或用户要求时下载。

## 训练目录

- 目录名使用 `<实验名>-<训练预算>-<Git 短提交号>-<YYYYMMDD-HHMMSS>`，时间戳使用北京时间。训练不同次启动使用各自独立目录。
- 启动脚本、启动配置、提交号、日志和退出状态放在训练目录根下。实际生效的配置、指标、checkpoint 和导出的策略放在 `training/`。
- 本地调试短跑使用独立的 `output/smoke-<YYYYMMDD-HHMMSS>/`。

## 评测目录

- 路径为 `output/<训练目录>/evaluation/iteration-<checkpoint 轮次>/<套件>-<后端>-<机器>-<YYYYMMDD-HHMMSS>/`。同一 checkpoint 的每次评测都新建目录。
- 评测根目录保存 `eval-config.json` 和 `summary.json`，记录实际使用的权重、种子、物理参数、后端和执行机器。指标和视频分开生成时，分别记录各自的执行机器。
- 每个评测条件使用独立子目录，如 `nominal/`，逐项报告保存为 `assessment.json`，有录制时将视频保存为 `replay.mp4`。

## 清理

- 训练期间按训练配置保存 checkpoint。用户要求清理中间 checkpoint 时，每轮保留最新完整 checkpoint、导出的 `policy.pt`，以及已指定用于对照的权重。配置、日志、指标和评测产物随对应训练目录保存。
