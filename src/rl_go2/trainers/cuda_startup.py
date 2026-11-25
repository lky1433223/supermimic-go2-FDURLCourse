"""Reserve a CUDA driver pool for reuse throughout training."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import ctypes
import math
import os

import torch


def _cuda_call(operation: Callable[..., tuple], *args: object) -> tuple:
    status, *values = operation(*args)
    if status.value:
        raise RuntimeError(f"{operation.__name__} failed: {status.name}")
    return tuple(values)


@contextmanager
def reserve_startup_memory(size_gib: float) -> Iterator[None]:
    """Prime a process-local device pool and keep it current until training ends."""
    if type(size_gib) not in (int, float) or not math.isfinite(size_gib) or size_gib < 0:
        raise ValueError("startup_gpu_reserve_gib must be finite and nonnegative")
    if not size_gib:
        yield
        return

    device = torch.device("cuda", torch.cuda.current_device())
    gib = 1024 ** 3
    free, total = torch.cuda.mem_get_info(device)
    if size_gib > free / gib:
        raise RuntimeError(
            f"Cannot reserve {size_gib:g} GiB on {device}: "
            f"{free / gib:.2f} of {total / gib:.2f} GiB free"
        )

    from cuda.bindings import driver

    (previous_pool,) = _cuda_call(driver.cuDeviceGetMemPool, device.index)
    size_bytes = max(1, int(size_gib * gib))
    threshold = ctypes.c_uint64(2**64 - 1)
    attributes = driver.CUmemPool_attribute
    properties = driver.CUmemPoolProps()
    properties.allocType = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    properties.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    properties.location.id = device.index
    (pool,) = _cuda_call(driver.cuMemPoolCreate, properties)
    try:
        _cuda_call(driver.cuMemPoolSetAttribute, pool, attributes.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                   ctypes.addressof(threshold))
        stream = torch.cuda.current_stream(device)
        handle = driver.CUstream(stream.cuda_stream)
        (allocation,) = _cuda_call(driver.cuMemAllocFromPoolAsync, size_bytes, pool, handle)
        _cuda_call(driver.cuMemFreeAsync, allocation, handle)
        stream.synchronize()
        (reserved,) = _cuda_call(driver.cuMemPoolGetAttribute, pool, attributes.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)
        if reserved < size_bytes:
            raise RuntimeError(f"CUDA pool retained only {reserved / gib:.2f} of {size_gib:g} GiB on {device}")
        # Genesis can trim the device's default pool. Keep the reservation in
        # a separate current pool so its async allocations can reuse it anyway.
        _cuda_call(driver.cuDeviceSetMemPool, device.index, pool)
        try:
            print(f"GPU startup pool retained {reserved / gib:.2f} GiB on {device}, PID {os.getpid()}", flush=True)
            yield
        finally:
            _cuda_call(driver.cuDeviceSetMemPool, device.index, previous_pool)
    finally:
        _cuda_call(driver.cuMemPoolDestroy, pool)
