#!/usr/bin/env python3
# coding: utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append("..")
import jax
import jax.interpreters.xla as _xla
_xla.DeviceArray = jax.Array
import jaxlib.xla_extension as _ext
_all_devices = jax.devices()                 
_default_dev = type(_all_devices[0])        
_ext.CpuDevice = _default_dev
_ext.GpuDevice = _default_dev
_ext.TpuDevice = _default_dev
import optax
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from jax import numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze
from evojax.util import get_params_format_fn

import random

from method.simpleGa_ku import initialize_population, mutation, elitism, selection, gSbx

seed = 1
pde = 'burgers1d'

if pde == 'burgers1d':
    from src.pde._1DBurgers import get_fitness, policy, train_task, PINNs
pop_size = 50
dim = policy.num_params
max_iters = 100000
loss_ls = []
t_training = []
runtime = 0.0
train_iters = 0
max_time = 60

model = PINNs()
key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
dummy = jnp.zeros((1, 2))
params = model.init(key1, dummy) # Initialization call
_, format_params_fn = get_params_format_fn(params)

def get_grad(params, train_iters):
    state = train_task._reset_fn(jax.random.PRNGKey(train_iters))
    action = model.apply(format_params_fn(params), state.obs)
    grad = train_task.grad_fn(action)
    
    return jnp.sign(grad)
get_grad = jax.jit(get_grad)
params = initialize_population(pop_size, dim, seed)
scores = get_fitness(params)
while train_iters < max_iters and runtime < max_time:
    start = time.time()        
    '''ga'''
    o_params = []
    for i in range(pop_size//2):
        idx = (train_iters + 1) * (i + 1)
        keys = jax.random.split(jax.random.PRNGKey(idx), 5) 
        p1, p2 = selection(params, scores, key=keys[0])
        sign1 = get_grad(p1, idx)
        sign2 = get_grad(p2, idx)
        c1, c2 = gSbx(p1, p2, sign1, sign2, key=keys[2])
        if random.uniform(0, 1) < 0.1:
            c1 = mutation(c1, key=keys[3])
            c2 = mutation(c2, key=keys[4])
        o_params.append(c1)
        o_params.append(c2)
    o_params = jnp.array(o_params)
    o_scores = get_fitness(o_params)
    params, scores = elitism(jnp.concatenate((params, o_params)), jnp.concatenate((scores, o_scores)), pop_size)

    avg_loss = np.mean(np.array(scores, copy=False))
    loss_ls.append(-avg_loss)
    elapsed = time.time() - start
    t_training.append(elapsed)
    runtime += elapsed
    train_iters += 1

    print(f"iter={train_iters:5d}  time={runtime:6.2f}s  loss={loss_ls[-1]:.2e}")

# Print final stats
print(f"\nFinished at iter={train_iters}, last loss={loss_ls[-1]:.2e}, best loss={min(loss_ls):.2e}")

log_message = f"pde:{pde}, method:CMAES, Finished at iter={train_iters}, last loss={loss_ls[-1]:.2e}, best loss={min(loss_ls):.2e}"
with open('log/record.txt', "a") as log_file:
    log_file.write(log_message)
print(log_message)

# --- Reconstruct best parameters ---
flat_best = params[jnp.argmax(scores)]
this_dict = policy._format_params_fn(flat_best)
new_dict = unfreeze(this_dict)
for m in new_dict:
    for p in new_dict[m]:
        for k in new_dict[m][p]:
            new_dict[m][p][k] = new_dict[m][p][k][0]
new_dict = freeze(new_dict)

# --- Load data and prepare batches ---
# sim = pd.read_csv('linear.csv')
# sim = sim[sim.x <= 4.5]
# X = np.vstack([sim.x.values, sim.t.values]).T
# Y = sim[['u']].values

# # --- PINN prediction ---
# model = PINNs()
# pred = model.apply(new_dict, X)
# u = pred[:, 0:1]

# # --- Plotting ---
# u_viz = u.reshape(201, 193).T
# u_true = Y.reshape(201, 193).T
# ext = [0, 2, -1.5, 4.5]

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
# im1 = ax1.imshow(u_viz, origin='lower', extent=ext,
#                     interpolation='bilinear', cmap='rainbow', aspect=0.25)
# ax1.set(xlabel='t', ylabel='x', title='PINN Solution')
# plt.colorbar(im1, ax=ax1)

# im2 = ax2.imshow(u_true, origin='lower', extent=ext,
#                     interpolation='bilinear', cmap='rainbow', aspect=0.25)
# ax2.set(xlabel='t', ylabel='x', title='Simulated Solution')
# plt.colorbar(im2, ax=ax2)

# plt.tight_layout()
# plt.show()

# mse = jnp.mean((u_viz - u_true) ** 2)
# print(f"MSE = {mse:.2e}")




