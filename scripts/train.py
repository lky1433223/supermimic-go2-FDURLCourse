"""Train a Go2 policy and save the run configuration and artifacts."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

from rl_go2.config import load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/flat_ppo.yaml"))
    parser.add_argument("--backend", choices=["genesis", "mujoco", "isaac"])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--course", help="Train only this named course with its inherited settings")
    parser.add_argument("--gpu-reserve-gib", type=float,
                        help="Override startup GPU reservation for Genesis/CUDA SuperMimic (0 disables it)")
    parser.add_argument("--output", type=Path, help="Run root; model artifacts are written under training/")
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize training weights from a checkpoint")
    parser.add_argument("--driver-policy", type=Path, help="Optional TorchScript data driver for SuperMimic")
    args = parser.parse_args()
    config = load_experiment(args.config)
    for key in ("backend", "device", "num_envs"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    trainer = config["trainer"]
    if args.gpu_reserve_gib is not None:
        if not math.isfinite(args.gpu_reserve_gib) or args.gpu_reserve_gib < 0:
            parser.error("--gpu-reserve-gib must be finite and nonnegative")
        if args.gpu_reserve_gib and (config["backend"], config["device"], trainer["name"]) != (
            "genesis", "cuda", "supermimic"
        ):
            parser.error("--gpu-reserve-gib requires Genesis/CUDA SuperMimic")
        config["startup_gpu_reserve_gib"] = args.gpu_reserve_gib
    if args.course is not None:
        course = next((stage for stage in config.get("courses", []) if stage["name"] == args.course), None)
        if course is None:
            parser.error(f"Unknown course: {args.course}")
        iterations = course["iterations"] if args.iterations is None else args.iterations
        config.update(course["settings"])
        config["courses"] = [{**course, "iterations": iterations, "end_iteration": iterations}]
        trainer["iterations"] = iterations
    elif args.iterations is not None:
        if config.get("courses") and args.iterations > trainer["iterations"]:
            parser.error("--iterations cannot exceed the configured course budget")
        trainer["iterations"] = args.iterations
    if config["num_envs"] < 1 or trainer["iterations"] < 1:
        parser.error("num-envs and iterations must be positive")
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.is_file():
            parser.error("--init-checkpoint requires an existing checkpoint")
        trainer["init_checkpoint"] = str(args.init_checkpoint.resolve())
    if args.driver_policy is not None:
        if trainer["name"] != "supermimic":
            parser.error("--driver-policy requires the SuperMimic trainer")
        trainer["driver_policy"] = str(args.driver_policy.resolve())
    compile_threads = config.get("inductor_compile_threads")
    if compile_threads is not None:
        if type(compile_threads) is not int or compile_threads < 1:
            parser.error("inductor_compile_threads must be a positive integer")
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
    if trainer["name"] == "supermimic":
        from rl_go2.trainers.supermimic import train
    elif trainer["name"] == "ppo":
        from rl_go2.trainers.ppo import train
    else:
        parser.error(f'Unknown trainer: {trainer["name"]}')

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    output = args.output or Path("output") / f'{args.config.stem}-{trainer["iterations"]}-{commit[:7]}-{timestamp}'
    output.mkdir(parents=True, exist_ok=False)
    (output / "commit.txt").write_text(commit + "\n")
    (output / "launch-config.json").write_text(json.dumps(config, indent=2) + "\n")
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    launcher = output / "launch.sh"
    launcher.write_text(f"#!/bin/sh\ncd {shlex.quote(str(Path.cwd()))} || exit 1\nexec {command}\n")
    launcher.chmod(0o755)
    log_path = output / "train.log"
    print(f"Training output: {output.resolve()}", flush=True)
    print(f"Training log: {log_path.resolve()}", flush=True)
    exit_code = 1
    try:
        with log_path.open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                train(config, output / "training")
            except KeyboardInterrupt:
                exit_code = 130
                traceback.print_exc()
            except Exception:
                traceback.print_exc()
            else:
                exit_code = 0
    finally:
        (output / "exit-code").write_text(f"{exit_code}\n")
    if exit_code:
        print(f"Training exited with status {exit_code}. See {log_path.resolve()}", file=sys.stderr)
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
