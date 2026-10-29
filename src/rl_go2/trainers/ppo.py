from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
import hashlib

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from rl_go2.env import QuadrupedEnv
from rl_go2.tasks import get_task_type


class RslRlEnv(VecEnv):
    """The only place that depends on RSL-RL's TensorDict/VecEnv convention."""

    def __init__(self, env):
        self.env = env
        self.num_envs, self.num_actions = env.num_envs, env.num_actions
        self.device, self.cfg = env.device, env.config
        self.max_episode_length = env.max_episode_length

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def get_observations(self):
        return TensorDict({"policy": self.env.observe()}, batch_size=[self.num_envs])

    def step(self, actions):
        observation, reward, done, extras = self.env.step(actions)
        return TensorDict({"policy": observation}, batch_size=[self.num_envs]), reward, done, extras


def make_runner(env, log_dir=None):
    return OnPolicyRunner(RslRlEnv(env), deepcopy(env.config["trainer"]["runner"]),
                          log_dir=str(log_dir) if log_dir is not None else None, device=str(env.device))


def save_run_spec(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    asset = next(e["file"] for e in config["scene"]["entities"] if e.get("robot"))
    meshes = sorted((Path(asset).parent / "assets").glob("*"))
    metadata = {
        "schema": "rl-go2.policy/v1", "quaternion_order": "wxyz", "world_up": "z", "units": "SI",
        "task": config["task"]["name"],
        "observation_terms": get_task_type(config["task"]["name"]).observation_terms, "control": config["robot"],
        "control_dt": config["scene"]["sim_options"]["dt"] * config["robot"]["control_decimation"],
        "versions": {name: version(name) for name in ("torch", "genesis-world", "mujoco", "rsl-rl-lib")},
        "asset_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(asset), *meshes]},
    }
    (output / "policy_spec.json").write_text(json.dumps(metadata, indent=2) + "\n")


def train(config, output):
    if config["backend"] != "genesis":
        raise ValueError("PPO training currently targets Genesis; use MuJoCo for evaluation")
    torch.manual_seed(config["seed"])
    env = QuadrupedEnv(config)
    try:
        save_run_spec(config, output)
        runner = make_runner(env, output)
        runner.learn(config["trainer"]["iterations"], init_at_random_ep_len=True)
        runner.export_policy_to_jit(str(output), filename="policy.pt")
        (Path(output) / "curriculum.json").write_text(json.dumps(env.curriculum.state_dict(), indent=2) + "\n")
    finally:
        env.close()
