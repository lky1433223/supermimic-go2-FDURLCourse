import argparse
import json
from pathlib import Path

from rl_go2.config import load_experiment


def main():
    parser = argparse.ArgumentParser(description="Go2 training framework; smoke never updates policy weights")
    parser.add_argument("command", choices=["validate", "smoke", "train", "evaluate"])
    parser.add_argument("--config", default="configs/experiments/flat_ppo.yaml")
    parser.add_argument("--backend", choices=["genesis", "mujoco", "isaac"])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize actor/critic weights; start fresh optimizer and curriculum")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--check-runner", action="store_true", help="Initialize PPO and sample actions, without learning")
    args = parser.parse_args()
    if args.command == "evaluate":
        if args.run_dir is None:
            parser.error("evaluate requires --run-dir")
        config = json.loads((args.run_dir / "config.json").read_text())
        config.update(backend="mujoco", device="cpu", num_envs=1, curriculum={"name": "none"})
    else:
        config = load_experiment(args.config)
    for key in ("backend", "device", "num_envs"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.iterations is not None:
        config["trainer"]["iterations"] = args.iterations
    if args.init_checkpoint is not None:
        if args.command != 'train' or not args.init_checkpoint.is_file():
            parser.error('--init-checkpoint requires train and an existing checkpoint')
        config['trainer']['init_checkpoint'] = str(args.init_checkpoint.resolve())
    if config["num_envs"] < 1 or args.steps < 1 or config["trainer"]["iterations"] < 1:
        parser.error("num-envs, steps and iterations must be positive")
    if args.command == "validate":
        print(json.dumps(config, indent=2))
        return
    if args.command == "train":
        if args.output is None:
            parser.error("train requires a new --output directory")
        name = config["trainer"]["name"]
        if name == "supermimic":
            from rl_go2.trainers.supermimic import train
        elif name == "ppo":
            from rl_go2.trainers.ppo import train
        else:
            parser.error(f"Unknown trainer: {name}")
        train(config, args.output)
        return
    import torch
    from rl_go2.env import QuadrupedEnv
    torch.manual_seed(config["seed"])
    env = QuadrupedEnv(config, viewer=args.viewer, record_transitions=args.command == "smoke")
    try:
        runner = None
        policy = None
        if args.command == "evaluate":
            policy = torch.jit.load(str(args.run_dir / "policy.pt"), map_location=env.device).eval()
        elif args.check_runner:
            from rl_go2.trainers.ppo import make_runner
            runner = make_runner(env)
            policy = runner.get_inference_policy(device=str(env.device))
        returns = torch.zeros(env.num_envs, device=env.device)
        resets = 0
        with torch.inference_mode():
            for _ in range(args.steps):
                if runner:
                    action = policy(runner.env.get_observations())
                elif policy:
                    action = policy(env.observe())
                else:
                    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
                observation, reward, done, extras = env.step(action)
                if not torch.isfinite(observation).all() or not torch.isfinite(reward).all():
                    raise RuntimeError("Non-finite simulation output")
                returns += reward
                resets += int(done.sum())
        print(json.dumps({"backend": config["backend"], "device": str(env.device), "num_envs": env.num_envs,
                          "steps": args.steps, "observation_shape": list(observation.shape),
                          "mean_total_reward": returns.mean().item(), "episode_resets": resets,
                          "policy_updates": 0}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
