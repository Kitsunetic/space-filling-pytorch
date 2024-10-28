import torch as th
import triton
import triton.language as tl
from torch import Tensor

__all__ = ["encode_hilbert", "encode_hilbert_unpadded"]


@triton.jit
def _calculate_hilbert_distance(fx, fy, fz, space_size):
    x = ((fx + 1) / 2 * space_size).to(tl.int64)
    y = ((fy + 1) / 2 * space_size).to(tl.int64)
    z = ((fz + 1) / 2 * space_size).to(tl.int64)
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
    for i in tl.static_range(0, 16):
        q = 1 << i
        ret |= (x & q) << (2 * i + 2)
        ret |= (y & q) << (2 * i + 1)
        ret |= (z & q) << (2 * i + 0)

    return ret


@triton.jit
def _encode_hilbert_kernel(
    xyz_ptr,
    code_ptr,
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
    ret = _calculate_hilbert_distance(fx, fy, fz, space_size)

    if ASSIGN_BATCH_INDEX:
        ret |= pid_b.to(tl.int64) << 48

    tl.store(code_ptr + pid_b * N + offs_n, ret, mask=mask_n)


def encode_hilbert(
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

    code = xyz.new_empty(B, N, dtype=th.int64)
    grid = lambda meta: (B, triton.cdiv(N, meta["BLK"]))
    # BLK = max(32, min(2048, triton.next_power_of_2(N)))
    BLK = 1024
    _encode_hilbert_kernel[grid](
        xyz, code, B, N, space_size, x_offset, y_offset, z_offset, *xyz.stride(), BLK, assign_batch_index
    )
    return code


@triton.jit
def _encode_hilbert_unpadded_kernel(
    xyz_ptr,
    seqlen_ptr,
    code_ptr,
    space_size,
    x_offset,
    y_offset,
    z_offset,
    str_xyz_n,
    str_xyz_c,
    BLK: tl.constexpr,
    ASSIGN_BATCH_INDEX: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    i = tl.load(seqlen_ptr + pid_b)
    j = tl.load(seqlen_ptr + pid_b + 1)

    offs_n = i + pid_n * BLK + tl.arange(0, BLK)
    mask = offs_n < j
    xyz_ptrs = xyz_ptr + offs_n * str_xyz_n

    fx = tl.load(xyz_ptrs + x_offset * str_xyz_c, mask=mask)
    fy = tl.load(xyz_ptrs + y_offset * str_xyz_c, mask=mask)
    fz = tl.load(xyz_ptrs + z_offset * str_xyz_c, mask=mask)
    ret = _calculate_hilbert_distance(fx, fy, fz, space_size)

    if ASSIGN_BATCH_INDEX:
        ret |= pid_b.to(tl.int64) << 48

    tl.store(code_ptr + offs_n, ret, mask=mask)


def encode_hilbert_unpadded(
    xyz: Tensor,
    seqlen: Tensor,
    max_seqlen: int,
    space_size: int,
    convention="xyz",
    assign_batch_index=True,
):
    """Returns z-order code from given normalized point cloud
    Args:
        xyz (Tensor): N 3, float. Point cloud. Must be normalize into [-1, 1]
        seqlen (Tensor): b+1, int32
        max_seqlen (int):
        space_size (int): spatial resolution. Higher for fine, lower for coarse representation
        convention (str): xyz offset. Must be one of ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
        assign_batch_index (bool): whether to assign left 16bits for batch index
    Returns:
        distance (Tensor): N, int64
    """
    assert xyz.ndim == 2 and xyz.size(-1) == 3, xyz.shape
    convention = convention.lower()
    assert convention in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    B, N = len(seqlen) - 1, xyz.size(0)
    x_offset, y_offset, z_offset = convention.find("x"), convention.find("y"), convention.find("z")
    code = xyz.new_empty(N, dtype=th.int64)

    # BLK = max(32, min(2048, triton.next_power_of_2(max_seqlen)))
    BLK = 1024
    grid = (B, triton.cdiv(N, BLK))

    _encode_hilbert_unpadded_kernel[grid](
        xyz, seqlen, code, space_size, x_offset, y_offset, z_offset, *xyz.stride(), BLK, assign_batch_index
    )
    return code
