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
    assign_batch_index=False,
):
    """Returns z-order code from given normalized point cloud
    Args:
        xyz (Tensor): b n 3, float. Point cloud. Must be normalize into [-1, 1]
        space_size (int): spatial resolution. Higher for fine, lower for coarse representation
        convention (str): xyz offset. Must be one of ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
        assign_batch_index (bool): whether to assign first 15bits as batch index.
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
    BLK = max(32, min(4096, triton.next_power_of_2(N)))
    _encode_z_kernel[grid](
        xyz,
        distance,
        B,
        N,
        space_size,
        x_offset,
        y_offset,
        z_offset,
        *xyz.stride(),
        BLK=BLK,
        ASSIGN_BATCH_INDEX=assign_batch_index,
    )
    return distance


@triton.jit
def _encode_z_unpadded_kernel(
    xyz_ptr,
    seqlen_ptr,
    distance_ptr,
    space_size,
    x_offset,
    y_offset,
    z_offset,
    BLK: tl.constexpr,
    ASSIGN_BATCH_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    i = tl.load(seqlen_ptr + pid)
    j = tl.load(seqlen_ptr + pid + 1)

    offs_n = i + tl.arange(0, BLK)
    mask = offs_n < j
    xyz_ptrs = xyz_ptr + offs_n * 3
    fx = tl.load(xyz_ptrs + x_offset, mask=mask)
    fy = tl.load(xyz_ptrs + y_offset, mask=mask)
    fz = tl.load(xyz_ptrs + z_offset, mask=mask)
    ret = _calculate_zorder(fx, fy, fz, space_size)

    # assign batch index
    if ASSIGN_BATCH_INDEX:
        ret |= pid.to(tl.int64) << 48

    tl.store(distance_ptr + offs_n, ret, mask=mask)


def encode_z_unpadeded(
    xyz: Tensor,
    seqlen: Tensor,
    max_seqlen: int,
    space_size: int,
    convention="xyz",
    assign_batch_index=False,
) -> Tensor:
    """Returns z-order code from given normalized point cloud.
    Args:
        xyz (Tensor): b n 3, float. Point cloud. Must be normalize into [-1, 1]
        seqlen (Tensor): b+1, int32
        max_seqlen (int):
        space_size (int): spatial resolution. Higher for fine, lower for coarse representation
        convention (str): xyz offset. Must be one of ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
        assign_batch_index (bool): whether to assign first 15bits as batch index.
    Returns:
        distance (Tensor): N, int64
    """
    assert xyz.ndim == 2 and xyz.size(-1) == 3, xyz.shape
    assert xyz.is_contiguous()
    assert isinstance(max_seqlen, int)
    convention = convention.lower()
    assert convention in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    B, N = len(seqlen) - 1, xyz.size(0)
    x_offset, y_offset, z_offset = convention.find("x"), convention.find("y"), convention.find("z")

    distance = xyz.new_empty(N, dtype=th.int64)
    _encode_z_unpadded_kernel[(B,)](
        xyz,
        seqlen,
        distance,
        space_size,
        x_offset,
        y_offset,
        z_offset,
        BLK=triton.next_power_of_2(max_seqlen),
        ASSIGN_BATCH_INDEX=assign_batch_index,
    )
    return distance
