# Go2 实验看板

看板运行在 ICC，按需读取 `/home/lky/RL-Go2/output/`。在 Mac 上通过 SSH 转发打开页面。看板读取实验产物，按需转码结果写入独立缓存，浏览器只记录各条回放的播放位置。

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
- 点击后才请求视频。默认 HLS 支持按网速切档和拖动；原版 MP4 使用 HTTP Range。SSH 掉线后恢复隧道，再用页面的重连按钮继续。
- 选择自动、省流或清晰时，首次请求会转码当前需要的分片。可随时切换到原版 MP4。

## HLS 回放

ICC 需要在 PATH 中提供 `ffmpeg` 和 `ffprobe`。直接点击播放即可按需转码，每片 4 秒，准备好第一片后开始播放。播放器只预取附近分片，拖到未缓存的位置会直接从该位置转码。首次播放或跳转时，等待时间取决于服务器 CPU 和网络速度。

看板使用请求处理线程运行 FFmpeg，每次最多一个转码进程，解码限制两个线程，编码使用八个线程。同一分片并发请求会复用结果，转码失败会在播放器提示重试，也可切回原版。单次转码有超时限制，失败临时文件会清理。运行 Uvicorn 时使用默认的单 worker，服务停止后无需另行维护转码任务。

缓存默认放在 ICC 的 `/home/lky/.cache/rl-go2-dashboard/hls/on-demand-v1/`。遵循 `XDG_CACHE_HOME`，也可用 `GO2_DASHBOARD_CACHE` 指定其他磁盘位置。只缓存实际请求的分片，两档分别为最高 360p、400 kbit/s 和最高 720p、1400 kbit/s。

按需缓存默认限制为 2048 MiB，可用 `GO2_DASHBOARD_CACHE_MB` 设置正整数。每次提供分片后检查上限，超限时删除较久未使用的分片。转码期间可能暂时超过上限一个分片。源视频路径、大小或修改时间发生变化时，会使用新缓存，旧分片也参与淘汰。少量目录和锁文件会保留。

Mac 只使用浏览器缓冲和播放时间记录。看板通过 `Cache-Control: no-store` 避免 HTTP 分片持久缓存，服务器缓存上限与 Mac 浏览器内存缓冲分别管理。

之前通过 `python -m dashboard.hls <replay.mp4>` 完整转好的缓存也能继续播放。该手动入口生成的完整缓存位于缓存根目录，需手动清理，不计入按需分片的容量限制。

## 自动检查

从仓库根目录运行。测试会创建临时产物，覆盖嵌套场景、评测进行中、旧版目录和媒体范围请求。

```bash
uv pip install --python dashboard/.venv/bin/python httpx
dashboard/.venv/bin/python -m unittest discover -s dashboard/tests -v
```
