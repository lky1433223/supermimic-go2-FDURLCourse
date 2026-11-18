"""Check dependencies, validate an experiment, or run a simulation smoke test."""

import argparse
import json
import platform
from importlib.metadata import version


def check_dependencies():
    import genesis as gs
    import mujoco
    import torch
    from rsl_rl.runners import OnPolicyRunner

    print(f"Python {platform.python_version()} ({platform.machine()})")
    for package in ("numpy", "mujoco", "torch", "genesis-world", "rsl-rl-lib"):
        print(f"{package}: {version(package)}")

    assert OnPolicyRunner is not None
    assert torch.isfinite(torch.ones(2, 2) @ torch.ones(2, 2)).all()
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")
    print(f"PyTorch MPS: {torch.backends.mps.is_available()}")

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time > 0 and data.qvel[2] < 0
    print("MuJoCo headless step: OK")
    print(f"Genesis import: OK ({gs.__name__})")


def smoke(config, args):
    import torch
    from rl_go2.env import QuadrupedEnv

    torch.manual_seed(config["seed"])
    env = QuadrupedEnv(config, viewer=args.viewer, record_transitions=True)
    try:
        runner = None
        policy = None
        if args.check_runner:
            if config["trainer"]["name"] == "supermimic":
                from rl_go2.trainers.supermimic import Actor, TrainConfig
                cfg = TrainConfig(**config["trainer"])
                cfg.validate()
                policy = Actor(env.observe().shape[-1], env.num_actions, cfg.actor_hidden,
                               config["robot"]["action_clip"]).to(env.device)
            else:
                from rl_go2.trainers.ppo import make_runner
                runner = make_runner(env)
                policy = runner.get_inference_policy(device=str(env.device))

        returns = torch.zeros(env.num_envs, device=env.device)
        resets = 0
        with torch.inference_mode():
            for _ in range(args.steps):
                if runner is not None:
                    action = policy(runner.env.get_observations())
                elif policy is not None:
                    action = policy(env.observe())
                else:
                    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
                observation, reward, done, _ = env.step(action)
                if not torch.isfinite(observation).all() or not torch.isfinite(reward).all():
                    raise RuntimeError("Non-finite simulation output")
                returns += reward
                resets += int(done.sum())
        return {
            "backend": config["backend"], "device": str(env.device), "num_envs": env.num_envs,
            "steps": args.steps, "observation_shape": list(observation.shape),
            "mean_total_reward": returns.mean().item(), "episode_resets": resets, "policy_updates": 0,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["dependencies", "validate", "smoke"],
                        default="dependencies")
    parser.add_argument("--config", default="configs/experiments/flat_ppo.yaml")
    parser.add_argument("--backend", choices=["genesis", "mujoco", "isaac"])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--driver-policy", help="Optional TorchScript data driver for SuperMimic")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--check-runner", action="store_true", help="Sample policy actions without learning")
    args = parser.parse_args()
    if args.command == "dependencies":
        check_dependencies()
        return

    from pathlib import Path
    from rl_go2.config import load_experiment

    config = load_experiment(args.config)
    for key in ("backend", "device", "num_envs"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.iterations is not None:
        config["trainer"]["iterations"] = args.iterations
    if args.driver_policy is not None:
        if config["trainer"]["name"] != "supermimic":
            parser.error("--driver-policy requires the SuperMimic trainer")
        config["trainer"]["driver_policy"] = str(Path(args.driver_policy).resolve())
    if config["num_envs"] < 1 or args.steps < 1 or config["trainer"]["iterations"] < 1:
        parser.error("num-envs, steps and iterations must be positive")
    print(json.dumps(config if args.command == "validate" else smoke(config, args), indent=2))


if __name__ == "__main__":
    main()
