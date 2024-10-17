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
def point_to_hilbert_distance_3d_depth16_fp32_kernel(
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
    tl.store(distance_ptr + pid_b * N + offs_n, ret, mask=mask_n)


def point_to_hilbert_distance_3d_depth16_fp32(
    xyz: Tensor,
    space_size: int,
    convention="xyz",
):
    """Returns z-order code from given normalized point cloud.
    Args:
        xyz (Tensor): b n 3, int32. Point cloud. Must be normalize into [-1, 1].
        space_size (int): spatial resolution. Higher for fine, lower for coarse representation.
        convention (str): xyz offset. Must be one of ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    Returns:
        distance (Tensor): b n, int64
    """
    assert xyz.ndim == 3 and xyz.size(-1) == 3, xyz.shape
    convention = convention.lower()
    assert convention in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    B, N = xyz.shape[:2]
    x_offset, y_offset, z_offset = convention.find("x"), convention.find("y"), convention.find("z")

    distance = xyz.new_empty(B, N, dtype=th.int64)
    grid = lambda meta: (B, triton.cdiv(N, meta["BLOCK_SIZE"]))
    BLOCK_SIZE = max(32, min(4096, triton.next_power_of_2(N)))
    point_to_hilbert_distance_3d_depth16_fp32_kernel[grid](
        xyz, distance, B, N, space_size, x_offset, y_offset, z_offset, *xyz.stride(), BLOCK_SIZE=BLOCK_SIZE
    )
    return distance
