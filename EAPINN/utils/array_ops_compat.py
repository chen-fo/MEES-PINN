import numpy as np
import jax.numpy as jnp

def istensorlist(values):
    """判断列表中是否含有 JAX 数组（DeviceArray）。"""
    return any(isinstance(v, jnp.ndarray) for v in values)

def convert_to_array(value):
    """
    把 list/tuple of DeviceArray 变为拼接后的 DeviceArray，
    否则一律转成 jnp.array。
    """
    if isinstance(value, (list, tuple)) and istensorlist(value):
        return jnp.stack(value, axis=0)
    return jnp.array(value)

def hstack(tup):
    """
    类似 np.hstack，但优先用 JAX 拼接。
    """
    if isinstance(tup[0], jnp.ndarray):
        return jnp.hstack(tup)
    return np.hstack(tup)

def roll(a, shift, axis):
    """
    类似 np.roll，但支持 JAX 数组。
    """
    if isinstance(a, jnp.ndarray):
        return jnp.roll(a, shift, axis)
    return np.roll(a, shift, axis=axis)

def zero_padding(array, pad_width):
    """
    支持稀疏索引格式与 JAX/NumPy 数组的零填充。
    """
    # SparseTensor 格式仍保持原逻辑
    if isinstance(array, (list, tuple)) and len(array) == 3:
        indices, values, dense_shape = array
        indices = [(i + pad_width[0][0], j + pad_width[1][0]) for i, j in indices]
        dense_shape = (
            dense_shape[0] + sum(pad_width[0]),
            dense_shape[1] + sum(pad_width[1]),
        )
        return indices, values, dense_shape
    # JAX 数组
    if isinstance(array, jnp.ndarray):
        return jnp.pad(array, pad_width)
    # NumPy 数组
    return np.pad(array, pad_width)