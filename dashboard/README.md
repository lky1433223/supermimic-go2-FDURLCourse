# Go2 实验看板

看板运行在 ICC，按需读取 `/home/lky/RL-Go2/output/`。在 Mac 上通过 SSH 转发打开页面。看板只读取产物，浏览器只记录各条回放的播放位置。

浏览器入口是 Uvicorn 启动后的 HTTP 地址。`templates/*.jinja` 由服务器填入实验数据后渲染。

## 在 ICC 启动

将包含 `dashboard/` 的提交同步到 ICC 后，在仓库根目录运行以下命令。依赖安装在独立虚拟环境。

```bash
cd /home/lky/RL-Go2
uv venv dashboard/.venv
uv pip install --python dashboard/.venv/bin/python -r dashboard/requirements.txt
GO2_DASHBOARD_OUTPUT=/home/lky/RL-Go2/output \
  dashboard/.venv/bin/python -m uvicorn dashboard.app:create_app --factory --host 127.0.0.1 --port 8765
```

在 Mac 上另开一个终端维持隧道，然后访问 [本机的看板入口](http://127.0.0.1:8765/)。

```bash
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -N -L 127.0.0.1:8765:127.0.0.1:8765 ICC
```

## 产物和回放

- 支持 `dr-vx0-40k-a9a8e2b`、`point-push-40k-08af176` 和使用当前仓库目录约定的新实验。旧实验的目录仍可在列表中找到，缺少当前训练配置时会提示格式问题。
- 训练页读取 `training/config.json` 和 `training/metrics.jsonl`。评测按 checkpoint 和套件列出，单项页识别评测根目录、工况子目录、场景和工况两层子目录中的报告与回放。
- 点击后才请求视频。播放和拖动走 HTTP Range。SSH 掉线后恢复隧道，再用页面的重连按钮继续。
- 默认直接播放已有的 `replay.mp4`。若对应目录已提供 `replay-preview.mp4`，页面优先采用预览，并提供原版切换。看板只读取已有媒体文件。

## 本地验证

从仓库根目录运行。测试会创建临时产物，覆盖嵌套场景、评测进行中、旧版目录和媒体范围请求。

```bash
uv pip install --python dashboard/.venv/bin/python httpx
dashboard/.venv/bin/python -m unittest discover -s dashboard/tests -v
```
