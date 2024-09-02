from typing import Tuple

import torch as th
import triton
import triton.language as tl
from torch import Tensor


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_SIZE": 32}),
#         triton.Config({"BLOCK_SIZE": 64}),
#         triton.Config({"BLOCK_SIZE": 128}),
#         triton.Config({"BLOCK_SIZE": 256}),
#         triton.Config({"BLOCK_SIZE": 512}),
#         triton.Config({"BLOCK_SIZE": 1024}),
#     ],
#     key=["BN"],
# )
@triton.jit
def point_to_zorder_3d_depth16_fp32_kernel(
    xyz_ptr,
    distance_ptr,
    B,
    N,
    space_size,
    x_offset,
    y_offset,
    z_offset,
    str_xyz_B,
    str_xyz_N,
    str_xyz_C,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_n = offs_n < N

    xyz_ptrs = xyz_ptr + pid_b * str_xyz_B + offs_n * str_xyz_N
    fx = tl.load(xyz_ptrs + x_offset * str_xyz_C, mask=mask_n)
    fy = tl.load(xyz_ptrs + y_offset * str_xyz_C, mask=mask_n)
    fz = tl.load(xyz_ptrs + z_offset * str_xyz_C, mask=mask_n)
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
    tl.store(distance_ptr + pid_b * N + offs_n, ret, mask=mask_n)


def point_to_zorder_3d_depth16_fp32(
    xyz: Tensor,
    space_size: int,
    x_offset: int = 0,
    y_offset: int = 1,
    z_offset: int = 2,
):
    assert xyz.ndim == 3, xyz.shape
    assert xyz.size(-1) == 3, xyz.shape
    B, N = xyz.shape[:2]

    distance = xyz.new_empty(B, N, dtype=th.int64)
    grid = lambda meta: (B, triton.cdiv(N, meta["BLOCK_SIZE"]))
    BLOCK_SIZE = max(32, min(4096, triton.next_power_of_2(N)))
    point_to_zorder_3d_depth16_fp32_kernel[grid](
        xyz, distance, B, N, space_size, x_offset, y_offset, z_offset, *xyz.stride(), BLOCK_SIZE=BLOCK_SIZE
    )
    return distance
