"""Hold a temporary CUDA allocation during CPU-side training startup."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import math
import os

import torch


@contextmanager
def reserve_startup_memory(size_gib: float) -> Iterator[Callable[[], None] | None]:
    """Reserve memory on the current CUDA device until the simulator needs it."""
    if type(size_gib) not in (int, float) or not math.isfinite(size_gib) or size_gib < 0:
        raise ValueError("startup_gpu_reserve_gib must be finite and nonnegative")
    if not size_gib:
        yield None
        return

    device = torch.device("cuda", torch.cuda.current_device())
    gib = 1024 ** 3
    free, total = torch.cuda.mem_get_info(device)
    if size_gib > free / gib:
        raise RuntimeError(
            f"Cannot reserve {size_gib:g} GiB on {device}: "
            f"{free / gib:.2f} of {total / gib:.2f} GiB free"
        )
    allocation = torch.empty((int(size_gib * gib),), device=device, dtype=torch.uint8)

    def release() -> None:
        nonlocal allocation
        if allocation is not None:
            allocation = None
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            print(f"GPU startup reservation released on {device}", flush=True)

    try:
        print(f"GPU startup reservation: {size_gib:g} GiB on {device}, PID {os.getpid()}", flush=True)
        yield release
    finally:
        release()
