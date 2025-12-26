# 读取deepxde的数据构造IC曲线

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from .data_trans import DataLoader


class ICDataFitter:
    def __init__(self, datapath, input_dim, output_dim, t0=0.0, t_transpose=True):
        # 1. 加载数据
        loader = DataLoader()
        loader.load(datapath, input_dim=input_dim, output_dim=output_dim, t_transpose=t_transpose)
        data = loader.ref_data  # numpy.ndarray, shape (N, input_dim+output_dim)

        # 2. 分离输入和输出
        coords = data[:, :input_dim]  # e.g. [x, t] 或 [x, y, t]
        values = data[:, input_dim:]  # e.g. [u] 或 [u, v]

        # 3. 只保留 t == t0 的样本
        #    假设时间是最后一列
        eps = 1e-6
        mask = np.isclose(coords[:, -1], t0, atol=eps)
        pts = coords[mask, :-1]  # 空间坐标 (x) 或 (x,y)
        vals = values[mask].squeeze()  # 对应的 u 值

        # 4. 构造插值器
        self._interp = LinearNDInterpolator(pts, vals)

    def __call__(self, x):
        """
        在任意空间点 x 处评估初始场 u(x, t0)。
        x: numpy.ndarray, shape (M, D) 其中 D=input_dim-1
        返回: numpy.ndarray, shape (M,) 或 (M,1)
        """
        y = self._interp(x)
        # 对于插值器外推值会得到 nan，可按需处理
        return y

    def sample(self, mode="all", size=None, random_state=None):
        """
        可选的采样接口：返回所有点或随机子集。
        mode="all" -> 返回所有 pts
        mode="random" -> 返回 size 个随机采样
        """
        pts = self._interp.xi  # LinearNDInterpolator 存储在 xi
        vals = self._interp.values
        if mode == "all" or size is None:
            return pts, vals
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(pts), size, replace=False)
        return pts[idx], vals[idx]
