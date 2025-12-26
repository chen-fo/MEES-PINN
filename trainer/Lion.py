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

def train(get_fitness, policy, sim_mgr, pop_size=32, init_stdev=0.02, max_iters=5000, seed=0, lr=0.0001):
    """
    Lion Optimizer (NeurIPS 2023) - Google DeepMind
    本质：一种只使用梯度符号（Sign）的动量梯度下降方法。
    """
    key = jax.random.PRNGKey(seed)
    params = jax.random.normal(key, (policy.num_params,)) * init_stdev
    
    # Lion 推荐参数
    beta1 = 0.9
    beta2 = 0.99
    weight_decay = 0.01 
    
    # 动量状态 (Momentum)
    exp_avg = jnp.zeros_like(params)

    loss_ls = []
    various_loss_ls = []
    iter_time_ls = []
    runtime = 0.0
    train_iters = 0

    best_loss = np.inf
    best_flat_params = np.array(params, copy=True)

    # Loss 定义
    def loss_fn(p):
        # 构造 Batch 输入
        p_batched = jnp.repeat(p[None, :], pop_size, axis=0) 
        losses, scores = get_fitness(sim_mgr, p_batched)
        # 目标是最小化 Loss (-fitness)
        return -jnp.mean(scores), jnp.mean(losses, axis=0)

    # JAX 自动微分计算梯度
    grad_fn = jit(value_and_grad(loss_fn, has_aux=True))

    # Lion 更新公式
    @jit
    def update_step(params, exp_avg, grads, lr):
        # 1. 梯度裁剪 (防止 PINN 梯度爆炸)
        grad_norm = jnp.linalg.norm(grads)
        scale_factor = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-6))
        grads = grads * scale_factor
        
        # 2. Lion 核心公式 (依赖梯度 grads)
        # c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        update = jnp.sign(exp_avg * beta1 + grads * (1 - beta1))
        
        # w_t = w_{t-1} - lr * (sign(c_t) + lambda * w_{t-1})
        new_params = params - lr * (update + weight_decay * params)
        
        # m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
        new_exp_avg = exp_avg * beta2 + grads * (1 - beta2)
        
        return new_params, new_exp_avg

    print(f"Start Lion Training (Gradient-Based): batch_size={pop_size}, lr={lr}")

    while train_iters < max_iters:
        t0 = time.time()
        
        # 1. 反向传播计算梯度
        (loss_val, various_loss_val), grads = grad_fn(params)
        
        if jnp.isnan(loss_val):
            print(f"!!! Error: Loss is NaN at iter {train_iters}. Stopping.")
            break

        # 2. 应用梯度更新
        params, exp_avg = update_step(params, exp_avg, grads, lr)

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

    print(f"\nFinished Lion at iter={train_iters}, best loss={best_loss:.2e}")

    return Result(
        best_w=best_flat_params, 
        best_fit=-best_loss, 
        evals=train_iters * pop_size, 
        iter_time_ls=iter_time_ls, 
        loss_ls=loss_ls, 
        various_loss_ls=various_loss_ls
    )