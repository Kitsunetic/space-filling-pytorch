from typing import Tuple

import torch as th
import triton
import triton.language as tl
from torch import Tensor

__all__ = ["encode_z", "encode_z_unpadeded"]


@triton.jit
def _calculate_zorder(fx, fy, fz, space_size):
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

    return ret


@triton.jit
def _encode_z_kernel(
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
    BLK: tl.constexpr,
    ASSIGN_BATCH_INDEX: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLK + tl.arange(0, BLK)
    mask_n = offs_n < N

    xyz_ptrs = xyz_ptr + pid_b * str_xyz_B + offs_n * str_xyz_N
    fx = tl.load(xyz_ptrs + x_offset * str_xyz_C, mask=mask_n)
    fy = tl.load(xyz_ptrs + y_offset * str_xyz_C, mask=mask_n)
    fz = tl.load(xyz_ptrs + z_offset * str_xyz_C, mask=mask_n)
    ret = _calculate_zorder(fx, fy, fz, space_size)

    # assign batch index
    if ASSIGN_BATCH_INDEX:
        ret |= pid_b.to(tl.int64) << 48

    tl.store(distance_ptr + pid_b * N + offs_n, ret, mask=mask_n)


def encode_z(
    xyz: Tensor,
    space_size: int,
    convention="xyz",
    assign_batch_index=True,
):
    """Returns z-order code from given normalized point cloud
    Args:
        xyz (Tensor): b n 3, float. Point cloud. Must be normalize into [-1, 1]
        space_size (int): spatial resolution. Higher for fine, lower for coarse representation
        convention (str): xyz offset. Must be one of ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
        assign_batch_index (bool): whether to assign left 16bits for batch index
    Returns:
        distance (Tensor): b n, int64
    """
    assert xyz.ndim == 3 and xyz.size(-1) == 3, xyz.shape
    convention = convention.lower()
    assert convention in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    B, N = xyz.shape[:2]
    x_offset, y_offset, z_offset = convention.find("x"), convention.find("y"), convention.find("z")

    distance = xyz.new_empty(B, N, dtype=th.int64)
    grid = lambda meta: (B, triton.cdiv(N, meta["BLK"]))
    BLK = max(32, min(2048, triton.next_power_of_2(N)))
    _encode_z_kernel[grid](
        xyz, distance, B, N, space_size, x_offset, y_offset, z_offset, *xyz.stride(), BLK, assign_batch_index
    )
    return distance


@triton.jit
def _encode_z_unpadded_kernel(
    xyz_ptr,
    batch_idx_ptr,
    code_ptr,
    space_size,
    x_offset,
    y_offset,
    z_offset,
    str_xyz_n,
    str_xyz_c,
    N,
    BLK: tl.constexpr,
    ASSIGN_BATCH_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLK + tl.arange(0, BLK)
    mask = offs_n < N
    xyz_ptrs = xyz_ptr + offs_n * str_xyz_n
    fx = tl.load(xyz_ptrs + x_offset * str_xyz_c, mask=mask)
    fy = tl.load(xyz_ptrs + y_offset * str_xyz_c, mask=mask)
    fz = tl.load(xyz_ptrs + z_offset * str_xyz_c, mask=mask)
    ret = _calculate_zorder(fx, fy, fz, space_size)

    if ASSIGN_BATCH_INDEX:
        batch_idx_ptrs = batch_idx_ptr + offs_n
        batch_idx = tl.load(batch_idx_ptrs, mask=mask).to(tl.int64)
        ret |= batch_idx << 48

    code_ptrs = code_ptr + offs_n
    tl.store(code_ptrs, ret, mask=mask)


def encode_z_unpadeded(
    xyz: Tensor,
    batch_idx: Tensor,
    space_size: int,
    convention="xyz",
    assign_batch_index=True,
) -> Tensor:
    """Returns z-order code from given normalized point cloud.
    Args:
        xyz (Tensor): N 3, float. Point cloud. Must be normalize into [-1, 1]
        batch_idx (Tensor): N, int(32 or 64).
        space_size (int): spatial resolution. Higher for fine, lower for coarse representation
        convention (str): xyz offset. Must be one of ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
        assign_batch_index (bool): whether to assign left 16bits for batch index
    Returns:
        distance (Tensor): N, int64
    """
    assert xyz.ndim == 2 and xyz.size(-1) == 3, xyz.shape
    convention = convention.lower()
    assert convention in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    x_offset, y_offset, z_offset = convention.find("x"), convention.find("y"), convention.find("z")

    N = len(xyz)
    code = xyz.new_empty(N, dtype=th.int64)

    BLK = max(32, min(2048, triton.next_power_of_2(N)))
    grid = (triton.cdiv(N, BLK),)

    _encode_z_unpadded_kernel[grid](
        xyz, batch_idx, code, space_size, x_offset, y_offset, z_offset, *xyz.stride(), N, BLK, assign_batch_index
    )
    return code
