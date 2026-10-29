"""Check imports, a PyTorch operation, and a headless MuJoCo simulation step."""

import platform
from importlib.metadata import version

import genesis as gs
import mujoco
import torch
from rsl_rl.runners import OnPolicyRunner


def main():
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


if __name__ == "__main__":
    main()
