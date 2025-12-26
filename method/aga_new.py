# AGA_iter.py —— 按“平均 loss 最小”保存；固定迭代次数
import os, sys, time
PROJECT_ROOT = "/home/chenfanke/TaskPINN"
os.environ.pop("PYTHONPATH", None)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# --- imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze
from flax import serialization

# ===== PDE / 接口（只改这里）=====
from src.pde.Wave2D_LongTime import get_fitness, policy, train_task
pde = "Wave2D_LongTime"
# =================================

# ===== 统一保存目录 =====
PROJECT_ROOT = Path("/home/chenfanke/TaskPINN")
TARGET_BASE  = PROJECT_ROOT / "train" / "AGA"
LOSS_DIR     = TARGET_BASE / "loss_iters"
RESULT_DIR   = TARGET_BASE / "result"
LOSS_TIME    = TARGET_BASE / "loss_time_csv"
PARAMS_DIR   = TARGET_BASE / "params"
LOSS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOSS_TIME.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

# ===== 命名配置（方程名_方法_网络参数）=====
method_name = "AGA"
net_arch    = "4*8"

# ===== AGA 超参数 =====
seed         = 1
n            = 50
m            = int(policy.num_params)
elite_rate   = 0.05
max_iters    = 3000
lower, upper = -10.0, 10.0

# 自适应参数范围
cx_min, cx_max = 0.60, 0.95
pm_min, pm_max = 0.001, 0.02
t_min,  t_max  = 2, 4
mut_sigma_base = 0.10 * (upper - lower)

# --- 工具：展平向量 -> params_tree ---
def flat_to_params_tree(flat_vector):
    flat1 = jnp.array([flat_vector])  # (1, P)
    this_dict = policy.format_params_fn(flat1)
    new_dict = unfreeze(this_dict)
    for m_ in new_dict:
        for p_ in new_dict[m_]:
            for k_ in new_dict[m_][p_]:
                new_dict[m_][p_][k_] = new_dict[m_][p_][k_][0]
    return freeze(new_dict)

# --- 工具：保存最优参数 ---
def save_best_params(params_tree):
    filename = f"{pde}_{method_name}_{net_arch}.msgpack"
    path = PARAMS_DIR / filename
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        f.write(serialization.to_bytes(params_tree))
    os.replace(tmp_path, path)
    return path

# ===== AGA 主过程 =====
def aga_train(
    get_fitness_fn,
    n=50, m=260,
    elite_rate=0.05,
    seed=1,
    max_iters=5,
    lower=-10.0, upper=10.0,
):
    key = jax.random.PRNGKey(seed)
    key, k1 = jax.random.split(key)
    pop = jax.random.uniform(k1, shape=(n, m), minval=lower, maxval=upper)

    elite = max(1, int(n * elite_rate))

    no_improve = 0
    iter_time_ls = []
    loss_ls = []

    # === 初始评估 ===
    fitness = get_fitness_fn(pop)
    best_idx = int(jnp.argmax(fitness))
    best_fit = float(fitness[best_idx])
    best_vec_indiv = np.array(pop[best_idx])

    best_loss = np.inf
    saved_vec = None

    for it in range(1, max_iters + 1):
        t0 = time.time()

        # 种群分散度驱动自适应
        std = jnp.std(pop, axis=0)
        D = jnp.clip(std / (upper - lower + 1e-9), 0.0, 1.0).mean()
        cx_rate = float(cx_min + (cx_max - cx_min) * (1.0 - D))
        pm_base = float(pm_min + (pm_max - pm_min) * (1.0 - D))
        t = int(jnp.floor(t_min + (1.0 - D) * (t_max - t_min)))
        t = max(t_min, min(t, t_max))
        mut_sigma = mut_sigma_base

        if no_improve >= 10:
            pm_base = min(pm_base * 1.5, pm_max)
            no_improve = 0

        # 精英
        elite_idx = jnp.argsort(-fitness)[:elite]
        elites = pop[elite_idx]

        # 锦标赛选择
        def tournament_select(rng, pop, fitness, ksize):
            idx = jax.random.randint(rng, (ksize,), 0, pop.shape[0])
            return idx[jnp.argmax(fitness[idx])]

        children, parent_fits = [], []
        num_children = n - elite
        for _ in range(num_children):
            key, k1 = jax.random.split(key)
            key, k2 = jax.random.split(key)
            p1 = tournament_select(k1, pop, fitness, t)
            p2 = tournament_select(k2, pop, fitness, t)

            key, k3 = jax.random.split(key)
            key, k4 = jax.random.split(key)
            do_cx = jax.random.bernoulli(k3, cx_rate)
            mask = jax.random.bernoulli(k4, 0.5, shape=(m,))
            child = jnp.where(do_cx & mask, pop[p1], pop[p2])

            children.append(child)
            parent_fits.append((fitness[p1] + fitness[p2]) / 2.0)

        children = jnp.stack(children, axis=0)
        parent_fits = jnp.stack(parent_fits, axis=0)

        # 自适应变异率
        f_avg = float(jnp.mean(fitness))
        f_max = float(jnp.max(fitness))
        denom = (f_max - f_avg + 1e-8)
        pm_i = jnp.where(
            parent_fits >= f_avg,
            pm_min + (pm_max - pm_min) * (f_max - parent_fits) / denom,
            pm_max,
        )
        pm_i = jnp.maximum(pm_i, pm_base)

        key, km1 = jax.random.split(key)
        key, km2 = jax.random.split(key)
        mut_mask = jax.random.bernoulli(km1, pm_i[:, None], shape=children.shape)
        noise = jax.random.normal(km2, shape=children.shape) * mut_sigma
        children = jnp.clip(children + mut_mask * noise, lower, upper)

        # 新一代
        pop = jnp.vstack([elites, children])

        # 评估
        fitness = get_fitness_fn(pop)
        curr_idx = int(jnp.argmax(fitness))
        curr_fit = float(fitness[curr_idx])
        if curr_fit > best_fit + 1e-12:
            best_fit = curr_fit
            best_vec_indiv = np.array(pop[curr_idx])
            no_improve = 0
        else:
            no_improve += 1

        # === 平均 loss ===
        loss_iter = float(-jnp.mean(fitness))
        loss_ls.append(loss_iter)

        if loss_iter < best_loss:
            best_loss = loss_iter
            elite_mean = np.asarray(jnp.mean(elites, axis=0))
            saved_vec = elite_mean.copy()
            params_tree = flat_to_params_tree(saved_vec)
            save_path = save_best_params(params_tree)
            print(f"💾 [iter {it}] 新 best loss 保存：loss(avg)={best_loss:.4e} → {save_path}")

        elapsed = time.time() - t0
        iter_time_ls.append(elapsed)

        print(f"iter={it:5d}  time={np.sum(iter_time_ls):6.2f}s  loss(avg)={loss_iter:.2e}  best={best_loss:.2e}")

    if saved_vec is None:
        saved_vec = np.array(best_vec_indiv, copy=True)
    return saved_vec, best_fit, iter_time_ls, loss_ls

# ======= 训练 & 导出 =======
if __name__ == "__main__":
    best_vec, best_fit, iter_time_ls, loss_ls = aga_train(
        get_fitness_fn=get_fitness,
        n=n, m=m, elite_rate=elite_rate,
        seed=seed, max_iters=max_iters,
        lower=lower, upper=upper
    )

    # loss 曲线
    fig_path = LOSS_DIR / f"{pde}_loss_iter.png"
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_ls) + 1), loss_ls, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (avg over pop)')
    plt.title(f'{pde} Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 曲线已保存：{fig_path}")

    # 保存迭代-时间-损失
    iter_time_cumsum = np.cumsum(iter_time_ls)
    df_log = pd.DataFrame({
        "iter": np.arange(1, len(loss_ls) + 1),
        "cum_time": iter_time_cumsum,
        "loss": loss_ls
    })
    loss_time_csv_path = LOSS_TIME / f"{pde}_IterTime_Loss.csv"
    df_log.to_csv(loss_time_csv_path, index=False)
    print(f"✅ 累计耗时与损失已保存：{loss_time_csv_path}")

    # ===== 恢复最优参数 → 做预测 → 保存 result.csv =====
    flat_best = jnp.array([best_vec])
    this_dict = policy.format_params_fn(flat_best)
    new_dict = unfreeze(this_dict)
    for m_ in new_dict:
        for p_ in new_dict[m_]:
            for k_ in new_dict[m_][p_]:
                new_dict[m_][p_][k_] = new_dict[m_][p_][k_][0]
    params_tree = freeze(new_dict)

    X_input = train_task.X_candidate
    Y_true  = train_task.u_ref
    model   = train_task.net
    derivs  = model.derivatives(params_tree, X_input)

    # 适配 1/2/3 通道输出
    if Y_true.shape[1] == 1:
        u_pred = np.asarray(derivs['u'])
        df = pd.DataFrame({
            'x': X_input[:, 0],
            'y': X_input[:, 1] if X_input.shape[1] >= 2 else 0,
            't': X_input[:, 2] if X_input.shape[1] >= 3 else 0,
            'u_true': Y_true[:, 0],
            'u_pred': u_pred[:, 0],
        })
    elif Y_true.shape[1] == 2:
        u_pred = np.asarray(derivs['u'])
        v_pred = np.asarray(derivs['v'])
        df = pd.DataFrame({
            'x': X_input[:, 0],
            'y': X_input[:, 1],
            't': X_input[:, 2] if X_input.shape[1] == 3 else 0,
            'u_true': Y_true[:, 0],
            'v_true': Y_true[:, 1],
            'u_pred': u_pred[:, 0],
            'v_pred': v_pred[:, 0],
        })
    elif Y_true.shape[1] == 3:
        u_pred = np.asarray(derivs['u'])
        v_pred = np.asarray(derivs['v'])
        p_pred = np.asarray(derivs['p'])
        df = pd.DataFrame({
            'x': X_input[:, 0],
            'y': X_input[:, 1],
            't': X_input[:, 2],
            'u_true': Y_true[:, 0],
            'v_true': Y_true[:, 1],
            'p_true': Y_true[:, 2],
            'u_pred': u_pred[:, 0],
            'v_pred': v_pred[:, 0],
            'p_pred': p_pred[:, 0],
        })
    else:
        raise ValueError(f"Unsupported output dimension: {Y_true.shape[1]}")

    csv_path = RESULT_DIR / f"{pde}_Result.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ 结果数据已保存：{csv_path}")

    final_param_file = PARAMS_DIR / f"{pde}_{method_name}_{net_arch}.msgpack"
    print(f"🎯 训练最优参数（按平均loss）已存为：{final_param_file}")

