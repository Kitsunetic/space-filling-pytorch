import random

METHODS = ["hilbert", "z"]
CONVENTIONS = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]


def sample_random_method():
    return random.choice(METHODS), random.choice(CONVENTIONS)
