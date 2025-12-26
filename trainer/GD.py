import time
import os
import jax
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, jit
from flax.struct import dataclass
from typing import List

# 保持 Result 定义不变
@dataclass
class Result:
    best_w: jnp.ndarray
    best_fit: float
    evals: int
    iter_time_ls: List[float]
    loss_ls: List[float]
    various_loss_ls: List[float]

def train(get_fitness, policy, sim_mgr, pop_size=None, init_stdev=0.02, max_iters=5000, seed=0, lr=0.01):
    """
    Gradient Descent implementation.
    注意：为了计算梯度，get_fitness 必须是可微分的 JAX 函数。
    """
    # 1. 初始化参数
    key = jax.random.PRNGKey(seed)
    # 假设 policy.num_params 返回参数数量，我们初始化为扁平向量
    params = jax.random.normal(key, (policy.num_params,)) * init_stdev
    
    loss_ls = []
    various_loss_ls = []
    iter_time_ls = []
    runtime = 0.0
    train_iters = 0

    best_loss = np.inf
    best_flat_params = None
    best_params_history = []

    # 2. 定义 Loss Wrapper
    # 原 get_fitness 接受 (pop_size, param_dim)，GD 输入为 (1, param_dim)
    def loss_fn(p):
        p_reshaped = p[None, :] # Reshape to (1, num_params)
        losses, scores = get_fitness(sim_mgr, p_reshaped)
        # 目标是最小化 Loss，即最大化 Fitness (scores)
        # 通常 scores = -loss，所以这里我们要最小化 -mean(scores)
        total_loss = -jnp.mean(scores)
        
        # 返回 loss 和辅助数据 (various_loss_mean)
        return total_loss, jnp.mean(losses, axis=0)

    # JIT 编译梯度函数以加速
    grad_fn = jit(value_and_grad(loss_fn, has_aux=True))

    print(f"Start GD Training: lr={lr}")

    while train_iters < max_iters:
        t0 = time.time()
        
        # 3. 计算梯度和 Loss
        (loss_val, various_loss_val), grads = grad_fn(params)
        
        # 4. 参数更新 (Standard GD)
        params = params - lr * grads

        # Record solution of current iteration
        best_params_history.append(np.array(params, copy=True))

        # 记录数据
        loss_scalar = float(loss_val)
        loss_ls.append(loss_scalar)
        
        # various_loss 转换为 numpy 用于存储
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

    print(f"\nFinished GD at iter={train_iters}, best loss={best_loss:.2e}")

    # Save all historical solutions to a single file
    save_dir = "result/gd/solution"
    os.makedirs(save_dir, exist_ok=True)
    # Filename format: total_iters_timestamp.npy
    filename = f"{train_iters}_{time.strftime('%Y%m%d_%H%M%S')}.npy"
    save_path = os.path.join(save_dir, filename)
    np.save(save_path, np.array(best_params_history))
    print(f"Saved training history to {save_path}")

    return Result(
        best_w=best_flat_params,  
        best_fit=-best_loss, # 保持接口一致，返回 best_fit (即 -loss)
        evals=train_iters,   # GD 每次迭代计为1次评估
        iter_time_ls=iter_time_ls, 
        loss_ls=loss_ls, 
        various_loss_ls=various_loss_ls
    )