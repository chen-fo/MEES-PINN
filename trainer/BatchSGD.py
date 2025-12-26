import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit
from flax.struct import dataclass
from typing import List

@dataclass
class Result:
    best_w: jnp.ndarray
    best_fit: float
    evals: int
    iter_time_ls: List[float]
    loss_ls: List[float]
    various_loss_ls: List[float]

def train(get_fitness, policy, sim_mgr, pop_size=32, init_stdev=0.02, max_iters=5000, seed=0, lr=0.01):
    """
    Mini-Batch SGD implementation.
    """
    key = jax.random.PRNGKey(seed)
    params = jax.random.normal(key, (policy.num_params,)) * init_stdev
    
    loss_ls = []
    various_loss_ls = []
    iter_time_ls = []
    runtime = 0.0
    train_iters = 0

    best_loss = np.inf
    best_flat_params = np.array(params, copy=True)

    # Batch SGD Loss Wrapper
    def loss_fn(p):
        p_batched = jnp.repeat(p[None, :], pop_size, axis=0) 
        losses, scores = get_fitness(sim_mgr, p_batched)
        total_loss = -jnp.mean(scores)
        return total_loss, jnp.mean(losses, axis=0)

    grad_fn = jit(value_and_grad(loss_fn, has_aux=True))

    print(f"Start Batch-SGD Training: batch_size={pop_size}, lr={lr}")

    while train_iters < max_iters:
        t0 = time.time()
        
        (loss_val, various_loss_val), grads = grad_fn(params)
        
        if jnp.isnan(loss_val):
            print(f"!!! Error: Loss is NaN at iter {train_iters}. Stopping.")
            break

        # 梯度裁剪 (Gradient Clipping)
        grad_norm = jnp.linalg.norm(grads)
        max_grad_norm = 1.0 
        scale_factor = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-6))
        grads = grads * scale_factor
        
        # 参数更新
        params = params - lr * grads

        loss_scalar = float(loss_val)
        loss_ls.append(loss_scalar)
        
        v_loss_np = np.array(various_loss_val, copy=True)
        various_loss_ls.append(v_loss_np)

        if loss_scalar < best_loss:
            best_loss = loss_scalar
            best_flat_params = np.array(params, copy=True)

        elapsed = time.time() - t0
        iter_time_ls.append(elapsed)
        runtime += elapsed
        train_iters += 1

        if train_iters % 100 == 0 or train_iters == 1:
             print(f"iter={train_iters:5d}  time={runtime:6.2f}s  loss={loss_ls[-1]:.2e}  "
                   f"pde={v_loss_np[0]:.2e} ic={v_loss_np[1]:.2e} bc={v_loss_np[2]:.2e} data={v_loss_np[3]:.2e}")

    # --- 新增：训练结束后的总结输出 ---
    print(f"\nFinished BatchSGD at iter={train_iters}, best loss={best_loss:.2e}")

    return Result(
        best_w=best_flat_params, 
        best_fit=-best_loss, 
        evals=train_iters * pop_size, 
        iter_time_ls=iter_time_ls, 
        loss_ls=loss_ls, 
        various_loss_ls=various_loss_ls
    )