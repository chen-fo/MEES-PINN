"""Initial conditions."""

__all__ = ["IC"]

import numpy as np
import jax
import jax.numpy as jnp
from .boundary_conditions import BC


class IC(BC):
    """Initial condition: enforce u(x, t0) = func(x, t0)."""

    def __init__(self, geom, func, on_initial, component=0):
        # 不调用 BC.__init__，而是手动设置需要的属性
        self.geom = geom
        # func: 初始函数，已通过 return_jax 包装
        self.func = func
        # 把你传进来的 on_initial 自动 vectorize
        self.on_initial = jax.vmap(on_initial, in_axes=(0, 0))
        self.component = component

    def filter(self, X):
        """
           向量化筛选初始时刻 t=t0 的点。
           X: array-like (N, D)
           return: jnp.ndarray of shape (N,), dtype=bool
        """
        # 1) NumPy array
        X_np = jnp.array(X)
        # 2) 先让几何体标记 t0：shape (N,)
        geom_mask = self.geom.on_initial(X_np)
        # 3) 交给用户 on_initial：也返回 (N,) 的 bool
        mask = self.on_initial(X_np, geom_mask)
        # 4) 转成 JAX 布尔向量
        return jnp.array(mask, dtype=jnp.bool_)

    def collocation_points(self, X):
        mask = self.filter(X)
        return X[mask]

    def error(self, pred_bc, X_bc):
        # u_pred - u_true
        u_pred = pred_bc[:, self.component:self.component + 1]
        u_true = self.func(X_bc)
        return u_pred - u_true
