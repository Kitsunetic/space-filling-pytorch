import torch as th
from torch import Tensor

from space_filling_pytorch import encode_hilbert, encode_hilbert_unpadded, encode_z, encode_z_unpadeded

__all__ = ["encode", "encode_unpadded"]


def encode(
    xyz: Tensor,
    space_size: int,
    method: str,
    convention: str,
    assign_batch_index=True,
):
    """
    Args:
        xyz (Tensor): b n 3, float. Point cloud. Must be normalize into [-1, 1]
        space_size (int): spatial resolution. Higher for fine, lower for coarse representation
        method (str): one of ["hilbert", "z"]
        convention (str): xyz offset. Must be one of ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
        assign_batch_index (bool): whether to assign first 15bits as batch index.
    Returns:
        distance (Tensor): b n, int64
    """
    method = method.lower()
    convention = convention.lower()
    return {"hilbert": encode_hilbert, "z": encode_z}[method](
        xyz,
        space_size,
        convention,
        assign_batch_index,
    )


def encode_unpadded(
    xyz: Tensor,
    seqlen: Tensor,
    max_seqlen: int,
    space_size: int,
    method: str,
    convention: str,
    assign_batch_index=True,
):
    """
    Args:
        xyz (Tensor): N 3, float. Point cloud. Must be normalize into [-1, 1]
        seqlen (Tensor): b+1, int32
        max_seqlen (int):
        space_size (int): spatial resolution. Higher for fine, lower for coarse representation
        method (str): one of ["hilbert", "z"]
        convention (str): xyz offset. Must be one of ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
        assign_batch_index (bool): whether to assign first 15bits as batch index.
    Returns:
        distance (Tensor): N, int64
    """
    method = method.lower()
    convention = convention.lower()
    return {"hilbert": encode_hilbert_unpadded, "z": encode_z_unpadeded}[method](
        xyz,
        seqlen,
        max_seqlen,
        space_size,
        convention,
        assign_batch_index,
    )
