"""Global seeding — the single gateway for all randomness in this project.

Any new source of randomness (library, simulator, sampler) MUST be seeded
here. Do not call random.seed / np.random.seed / torch.manual_seed elsewhere.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
