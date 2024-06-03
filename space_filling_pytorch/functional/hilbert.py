from typing import Tuple

import torch as th
import triton
import triton.language as tl
from torchtyping import TensorType


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}),
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["BN"],
)
@triton.jit
def point_to_hilbert_distance_3d_depth16_fp32_kernel(
    xyz_ptr,
    distance_ptr,
    BN,
    space_size,
    x_offset,
    y_offset,
    z_offset,
    BLOCK_SIZE: tl.constexpr,
):
    # load data
    pid = tl.program_id(0)
    idx_bn = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx_bn < BN
    # TODO no coalescing
    fx = tl.load(xyz_ptr + idx_bn * 3 + x_offset, mask=mask)
    fy = tl.load(xyz_ptr + idx_bn * 3 + y_offset, mask=mask)
    fz = tl.load(xyz_ptr + idx_bn * 3 + z_offset, mask=mask)
    x = ((fx + 1) / 2 * space_size).to(tl.uint32)
    y = ((fy + 1) / 2 * space_size).to(tl.uint32)
    z = ((fz + 1) / 2 * space_size).to(tl.uint32)
    x = tl.minimum(tl.maximum(x, 0), space_size - 1)
    y = tl.minimum(tl.maximum(y, 0), space_size - 1)
    z = tl.minimum(tl.maximum(z, 0), space_size - 1)

    # calculate hilbert distance
    for i in tl.static_range(15, 0, -1):
        q = 1 << i
        p = q - 1

        # dim = 0
        x ^= tl.where(x & q, p, 0)

        # dim = 1
        cond = y & q
        t = (x ^ y) & p
        x ^= tl.where(cond, p, t)
        y ^= tl.where(cond, 0, t)

        # dim = 2
        cond = z & q
        t = (x ^ z) & p
        x ^= tl.where(cond, p, t)
        z ^= tl.where(cond, 0, t)

    y ^= x
    z ^= y

    t = 0
    for i in tl.static_range(15, 0, -1):
        q = 1 << i
        t ^= tl.where(z & q, q - 1, 0)

    x ^= t
    y ^= t
    z ^= t

    # write results
    ret = 0
    ix = x.to(tl.int64)
    iy = y.to(tl.int64)
    iz = z.to(tl.int64)
    for i in tl.static_range(0, 16):
        q = 1 << i
        ret |= (ix & q) << (2 * i + 2)
        ret |= (iy & q) << (2 * i + 1)
        ret |= (iz & q) << (2 * i + 0)
    tl.store(distance_ptr + idx_bn, ret, mask=mask)


def point_to_hilbert_distance_3d_depth16_fp32(
    xyz: TensorType["b", "n", 3, th.float32],
    space_size: int,
    x_offset: int = 0,
    y_offset: int = 1,
    z_offset: int = 2,
):
    assert xyz.ndim == 3, xyz.shape
    assert xyz.size(-1) == 3, xyz.shape
    B, N = xyz.shape[:2]

    distance = xyz.new_empty(B, N, dtype=th.int64)
    grid = lambda meta: (triton.cdiv(B * N, meta["BLOCK_SIZE"]),)
    point_to_hilbert_distance_3d_depth16_fp32_kernel[grid](
        xyz, distance, B * N, space_size, x_offset, y_offset, z_offset
    )
    return distance
