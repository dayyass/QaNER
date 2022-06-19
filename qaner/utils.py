import random

import numpy as np
import torch


def set_global_seed(seed: int):
    """
    Set global seed for reproducibility.

    Args:
        seed (int): Seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
