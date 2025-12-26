import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from jax import numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze
from evojax.algo import CMA_ES_JAX
'''Config'''
seed = 1
pop_size = 50
init_stdev=0.01
max_iters = 100000
max_time = 60
pde = 'burgers1d'

'''Initialize'''
if pde == 'burgers1d':
    from src.pde.Burgers1D import get_fitness, policy

solver = CMA_ES_JAX(
    pop_size=pop_size,
    init_stdev=init_stdev,
    param_size=policy.num_params,
    seed=seed,
)

loss_ls = []
t_training = []
runtime = 0.0
train_iters = 0

'''Trianing loop'''
while train_iters < max_iters and runtime < max_time:
    start = time.time()
    params = solver.ask()
    
    scores = get_fitness(params)
    solver.tell(fitness=scores)

    avg_loss = np.mean(np.array(scores, copy=False))
    loss_ls.append(-avg_loss)
    elapsed = time.time() - start
    t_training.append(elapsed)
    runtime += elapsed
    train_iters += 1

    print(f"iter={train_iters:5d}  time={runtime:6.2f}s  loss={loss_ls[-1]:.2e}")

print(f"\nFinished at iter={train_iters}, last loss={loss_ls[-1]:.2e}, best loss={min(loss_ls):.2e}")

'''Reconstruct best parameters''' 
flat_best = jnp.array([solver.best_params])
this_dict = policy._format_params_fn(flat_best)
new_dict = unfreeze(this_dict)
for m in new_dict:
    for p in new_dict[m]:
        for k in new_dict[m][p]:
            new_dict[m][p][k] = new_dict[m][p][k][0]
new_dict = freeze(new_dict)

'''Load data and prepare batches''' 
sim = pd.read_csv('linear.csv')
sim = sim[sim.x <= 4.5]
X = np.vstack([sim.x.values, sim.t.values]).T
Y = sim[['u']].values

'''PINN prediction'''
# model = PINNs()
# pred = model.apply(new_dict, X)
# u = pred[:, 0:1]

# '''Plotting'''
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
# plt.savefig(f"CMAES+{pde}.png")
# plt.tight_layout()
# plt.show()

# mse = jnp.mean((u_viz - u_true) ** 2)
# print(f"MSE = {mse:.2e}")
