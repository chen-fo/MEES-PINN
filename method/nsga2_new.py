#!/usr/bin/env python3
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze
from flax import serialization

# ===== 路径配置 =====
PROJECT_ROOT = Path("/home/chenfanke/TaskPINN")
os.environ.pop("PYTHONPATH", None)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 指定 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

TARGET_BASE  = PROJECT_ROOT / "train" / "NSGA2"
LOSS_DIR     = TARGET_BASE / "loss_iters"
RESULT_DIR   = TARGET_BASE / "result"
LOSS_TIME    = TARGET_BASE / "loss_time_csv"
PARAMS_DIR   = TARGET_BASE / "params"
for d in [LOSS_DIR, RESULT_DIR, LOSS_TIME, PARAMS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===== PDE 配置（保留你的依赖） =====
pde = "Burgers1D"
from src.pde.Burgers1D import get_multi_fitness, policy, train_task

method_name = "NSGA2_new"
net_arch    = "4*8"

# ===== 工具函数 =====
def flat_to_params_tree(flat_vector):
    """把扁平向量恢复成 policy 要求的 params tree"""
    flat1 = jnp.array([flat_vector])
    this_dict = policy.format_params_fn(flat1)
    new_dict = unfreeze(this_dict)
    for m in new_dict:
        for p in new_dict[m]:
            for k in new_dict[m][p]:
                # 去除 batch 维
                new_dict[m][p][k] = new_dict[m][p][k][0]
    return freeze(new_dict)

def save_best_params(params_tree):
    """原子性写入最优 params（msgpack via flax.serialization）"""
    filename = f"{pde}_{method_name}_{net_arch}.msgpack"
    path = PARAMS_DIR / filename
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        f.write(serialization.to_bytes(params_tree))
    os.replace(tmp_path, path)
    return path

# ===== NSGA-II 核心函数（完整保留并使用 JAX/numpy） =====
def fast_non_dominated_ranks(objs: jnp.ndarray) -> jnp.ndarray:
    N = objs.shape[0]
    objs_p = objs[:, None, :]   # (N,1,M)
    objs_q = objs[None, :, :]   # (1,N,M)
    less_equal = jnp.all(objs_p <= objs_q, axis=2)     # (N,N)
    strictly_less = jnp.any(objs_p < objs_q, axis=2)   # (N,N)
    dominates = jnp.logical_and(less_equal, strictly_less)

    dom_count = jnp.sum(dominates, axis=0)
    ranks = -jnp.ones((N,), dtype=jnp.int32)

    current = jnp.where(dom_count == 0)[0]
    r = 0
    while current.size > 0:
        ranks = ranks.at[current].set(r)
        reduces = jnp.sum(dominates[current, :], axis=0)
        dom_count = dom_count - reduces
        mask_unranked = (ranks < 0)
        current = jnp.where(jnp.logical_and(dom_count == 0, mask_unranked))[0]
        r += 1
    return ranks

def crowding_distance_for_front(objs: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    F = idx.size
    if F == 0:
        return jnp.array([], dtype=objs.dtype)
    if F == 1:
        return jnp.array([jnp.inf], dtype=objs.dtype)

    M = objs.shape[1]
    cd = jnp.zeros((F,), dtype=objs.dtype)
    sub = objs[idx, :]  # (F, M)

    for m in range(M):
        order = jnp.argsort(sub[:, m])
        sorted_vals = sub[order, m]
        denom = sorted_vals[-1] - sorted_vals[0]
        contrib = jnp.zeros((F,)).at[jnp.array([0, F-1])].set(jnp.inf)
        if F > 2:
            diff = (sorted_vals[2:] - sorted_vals[:-2]) / (denom + 1e-12)
            contrib = contrib.at[1:-1].add(diff)
        inv_order = jnp.empty_like(order).at[order].set(jnp.arange(F))
        cd = cd + contrib[inv_order]
    return cd

def crowding_distance_all(objs: jnp.ndarray, ranks: jnp.ndarray) -> jnp.ndarray:
    N = objs.shape[0]
    cd = jnp.zeros((N,), dtype=objs.dtype)
    max_rank = int(jnp.max(ranks)) if N > 0 else -1
    for r in range(max_rank + 1):
        idx = jnp.where(ranks == r)[0]
        if idx.size == 0:
            continue
        cd_r = crowding_distance_for_front(objs, idx)
        cd = cd.at[idx].set(cd_r)
    return cd

def binary_tournament(key, ranks: jnp.ndarray, crowd: jnp.ndarray, num: int):
    N = ranks.shape[0]
    key, k = jax.random.split(key)
    cand = jax.random.randint(k, (num, 2), 0, N)  # (num,2)
    a, b = cand[:, 0], cand[:, 1]
    ra, rb = ranks[a], ranks[b]
    ca, cb = crowd[a], crowd[b]
    better_a = jnp.logical_or(ra < rb, jnp.logical_and(ra == rb, ca > cb))
    chosen = jnp.where(better_a, a, b)
    return key, chosen

def sbx_crossover(key, p1: jnp.ndarray, p2: jnp.ndarray,
                  pc=0.9, eta_c=15.0, lower=0.0, upper=1.0):
    B, L = p1.shape
    key, kprob, ku = jax.random.split(key, 3)
    do_cx = jax.random.bernoulli(kprob, pc, (B, 1))
    u = jax.random.uniform(ku, (B, L), minval=0.0, maxval=1.0)
    beta = jnp.where(u <= 0.5, (2.0*u)**(1.0/(eta_c+1.0)),
                              (1.0/(2.0*(1.0-u)))**(1.0/(eta_c+1.0)))
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    c1 = jnp.where(do_cx, c1, p1)
    c2 = jnp.where(do_cx, c2, p2)
    c1 = jnp.clip(c1, lower, upper)
    c2 = jnp.clip(c2, lower, upper)
    return key, c1, c2

def polynomial_mutation(key, X, pm=0.01, eta_m=20.0,
                        lower=-5.0, upper=5.0):
    N, L = X.shape
    key, kmask, kr = jax.random.split(key, 3)
    mask = jax.random.bernoulli(kmask, pm, X.shape)
    r = jax.random.uniform(kr, X.shape, minval=0.0, maxval=1.0)

    xl, xu = lower, upper
    delta1 = (X - xl) / (xu - xl + 1e-12)
    delta2 = (xu - X) / (xu - xl + 1e-12)
    mut_pow = 1.0 / (eta_m + 1.0)

    r_less = (r < 0.5)
    xy1 = 1.0 - delta1
    val1 = 2.0*r + (1.0 - 2.0*r) * (xy1 ** (eta_m + 1.0))
    deltaq1 = (val1 ** mut_pow) - 1.0

    xy2 = 1.0 - delta2
    val2 = 2.0*(1.0 - r) + 2.0*(r - 0.5) * (xy2 ** (eta_m + 1.0))
    deltaq2 = 1.0 - (val2 ** mut_pow)

    deltaq = jnp.where(r_less, deltaq1, deltaq2)
    X_mut = X + jnp.where(mask, deltaq*(xu - xl), 0.0)
    X_mut = jnp.clip(X_mut, xl, xu)
    return key, X_mut

def environmental_selection(pop, objs, N):
    ranks = fast_non_dominated_ranks(objs)  # (2N,)
    selected_mask = jnp.zeros((pop.shape[0],), dtype=bool)
    selected_cnt = 0
    max_rank = int(jnp.max(ranks))

    for r in range(max_rank + 1):
        idx = jnp.where(ranks == r)[0]
        sz = int(idx.size)
        if selected_cnt + sz <= N:
            selected_mask = selected_mask.at[idx].set(True)
            selected_cnt += sz
        else:
            K = N - selected_cnt
            if K > 0:
                cd = crowding_distance_for_front(objs, idx)
                _, top_pos = jax.lax.top_k(cd, K)
                chosen = idx[top_pos]
                selected_mask = selected_mask.at[chosen].set(True)
                selected_cnt += K
            break

    next_pop = pop[selected_mask]
    next_objs = objs[selected_mask]
    next_ranks = ranks[selected_mask]
    return next_pop, next_objs, next_ranks

# ===== NSGA-II 主训练入口（所有 gen 改为 iter） =====
def nsga2(get_fitness, n=50, m=260, generations=2000, seed=0,
          lower=-5.0, upper=5.0,
          pc=0.9, pm=None, eta_c=15.0, eta_m=20.0):
    if pm is None:
        pm = 1.0 / m

    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    pop = jax.random.uniform(k0, (n, m), minval=lower, maxval=upper)

    objs = -(get_fitness(pop))
    ranks = fast_non_dominated_ranks(objs)
    crowd = crowding_distance_all(objs, ranks)

    best_total_hist, iter_time_ls = [], []
    bestFitness = jnp.inf
    bestParams = None
    runtime = 0.0

    for iter in range(1, generations+1):
        t0 = time.time()

        key, parent_idx = binary_tournament(key, ranks, crowd, n)
        parents = pop[parent_idx]

        key, kperm = jax.random.split(key)
        perm = jax.random.permutation(kperm, n)
        p1 = parents[perm[0::2]]
        p2 = parents[perm[1::2]]

        key, c1, c2 = sbx_crossover(key, p1, p2, pc=pc, eta_c=eta_c, lower=lower, upper=upper)
        children = jnp.vstack([c1, c2])
        key, children = polynomial_mutation(key, children, pm=pm, eta_m=eta_m, lower=lower, upper=upper)

        objs_children = -(get_fitness(children))

        pop_comb = jnp.vstack([pop, children])
        objs_comb = jnp.vstack([objs, objs_children])
        pop, objs, _ = environmental_selection(pop_comb, objs_comb, n)

        ranks = fast_non_dominated_ranks(objs)
        crowd = crowding_distance_all(objs, ranks)

        sum_losses = jnp.sum(objs, axis=1)
        best_idx = int(jnp.argmin(sum_losses))
        best_losses = objs[best_idx]
        best_total = float(jnp.sum(best_losses))

        best_total_hist.append(best_total)
        took = time.time() - t0
        iter_time_ls.append(took)
        runtime += took

        if best_total < float(bestFitness):
            bestFitness = best_total
            bestParams = pop[best_idx]
            params_tree_best = flat_to_params_tree(bestParams)
            save_path = save_best_params(params_tree_best)
            print(f"💾 [iter {iter}] 新 best_total={best_total:.4e} 保存至 {save_path}")

        print(f"iter={iter:5d}  time={runtime:6.2f}s  best_total={best_total:.4e}")

    # 保存日志（iter, cum_time, best_total）
    iter_time_cumsum = np.cumsum(iter_time_ls)
    df_log = pd.DataFrame({
        "iter": np.arange(1, len(best_total_hist)+1, dtype=int),
        "cum_time": iter_time_cumsum,
        "best_total": best_total_hist
    })
    loss_time_csv_path = LOSS_TIME / f"{pde}_IterTime_Loss.csv"
    df_log.to_csv(loss_time_csv_path, index=False)
    print(f"✅ 累计耗时与损失已保存：{loss_time_csv_path}")

    # 保存 loss 曲线
    fig_path = LOSS_DIR / f"{pde}_loss_curve.png"
    plt.figure(figsize=(10,6))
    plt.plot(range(1,len(best_total_hist)+1), best_total_hist, 'b-', lw=2)
    plt.xlabel("Iteration"); plt.ylabel("Best Total Loss")
    plt.title(f"{pde} NSGA-II Loss Curve")
    plt.grid(True, ls="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"📈 曲线已保存：{fig_path}")

    return bestParams, bestFitness, best_total_hist, iter_time_ls

# ===== 训练入口 =====
if __name__ == '__main__':
    n = 50
    m = policy.num_params
    generations = 500  

    best_w, best_fit, loss_ls, iter_time_ls = nsga2(get_multi_fitness, n=n, m=m, generations=generations)

    # ===== 预测结果保存 =====
    # 如果训练过程中未更新 best_w（None），则用最后种群中最好的个体
    if best_w is None:
        # 需要重新评估当前 pop: 这里简单退出并提示（通常不会发生）
        raise RuntimeError("训练结束但未保存到 best_w；请检查 get_multi_fitness 的返回值格式。")

    flat_best = jnp.array([best_w])
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
            't': X_input[:, 2] if X_input.shape[1] == 3 else 0,
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
    print(f"✅ 预测结果已保存：{csv_path}")

    final_param_file = PARAMS_DIR / f"{pde}_{method_name}_{net_arch}.msgpack"
    print(f"🎯 训练最优参数已存为：{final_param_file}")
