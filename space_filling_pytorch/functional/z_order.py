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
def point_to_zorder_3d_depth16_fp32_kernel(
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
    x = ((fx + 1) / 2 * space_size).to(tl.int64)
    y = ((fy + 1) / 2 * space_size).to(tl.int64)
    z = ((fz + 1) / 2 * space_size).to(tl.int64)
    x = tl.minimum(tl.maximum(x, 0), space_size - 1)
    y = tl.minimum(tl.maximum(y, 0), space_size - 1)
    z = tl.minimum(tl.maximum(z, 0), space_size - 1)

    # calculate z-order
    ret = 0
    for i in tl.static_range(0, 16):
        q = 1 << i
        ret |= (x & q) << (2 * i + 2)
        ret |= (y & q) << (2 * i + 1)
        ret |= (z & q) << (2 * i + 0)

    # write results
    tl.store(distance_ptr + idx_bn, ret, mask=mask)


def point_to_zorder_3d_depth16_fp32(
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
    point_to_zorder_3d_depth16_fp32_kernel[grid](
        xyz, distance, B * N, space_size, x_offset, y_offset, z_offset
    )
    return distance
