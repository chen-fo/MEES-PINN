import random
import numpy as np
from .real import Real

# 只支持 NumPy，默认 float32
real = Real(32)

# 全局随机种子（仅 Python + NumPy）
random_seed = None
def set_random_seed(seed: int):
    global random_seed
    random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
