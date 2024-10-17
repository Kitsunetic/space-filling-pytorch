import torch as th
from torch import Tensor

from space_filling_pytorch import hilbert_encode, z_order_encode

__all__ = ["encode"]


def encode(xyz: Tensor, space_size: int, method: str, convention: str):
    """
    Args:
        xyz (Tensor): b n 3, float32
        space_size (int): spatial resolution. Higher for fine, lower for coarse representation
        method (str): one of ["hilbert", "z"]
        convention (str): xyz offset. Must be one of ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    Returns:
        distance (Tensor): b n, int64
    """
    method = method.lower()
    convention = convention.lower()
    return {"hilbert": hilbert_encode, "z": z_order_encode}[method](xyz, space_size, convention)
