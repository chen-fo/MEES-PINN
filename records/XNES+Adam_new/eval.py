import os
os.environ['CUDA_VISIBLE_DEVICES']  ='2'
import sys
from flax.serialization import msgpack_restore
sys.path.insert(0, 'home/chenfanke/TaskPINN')

import jax
from jax import numpy as jnp

from src.pde.Heat2D_LongTime import PINN, get_fitness, policy

env = 'Heat2D_LongTime_XNES+Adam_new_4*8'

with open(f'train/XNES+Adam_new/params/{env}.msgpack', 'rb') as f:
    parambyte = f.read()

pytree = msgpack_restore(parambyte)
param_shapes = jax.tree_util.tree_map(lambda x: x.shape, pytree)

leaves, _ = jax.tree_util.tree_flatten(pytree)

flat_params_list = [jnp.ravel(leaf) for leaf in leaves]
params = jnp.concatenate(flat_params_list)
params = params[jnp.newaxis, :]
scores = get_fitness(params)
print(scores)