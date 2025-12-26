import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from flax import serialization

# --- 项目路径配置 ---
PROJECT_ROOT = Path("/home/chenfanke/TaskPINN")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- PDE 选择 ---
pde = "GrayScottEquation"
from src.pde.GrayScottEquation import get_fitness, policy, train_task

# --- 保存目录 ---
METHOD_NAME = "MOEAD"
NET_ARCH    = "4*8"  # 按网络结构填写
TARGET_BASE = PROJECT_ROOT / "train" / METHOD_NAME
LOSS_DIR    = TARGET_BASE / "loss_iters"
RESULT_DIR  = TARGET_BASE / "result"
LOSS_TIME   = TARGET_BASE / "loss_time_csv"
PARAMS_DIR  = TARGET_BASE / "params"
for d in [LOSS_DIR, RESULT_DIR, LOSS_TIME, PARAMS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- MOEAD 内部算子 ---
def sbx_crossover(key, p1, p2, pc=0.9, eta_c=15.0, lower=0.0, upper=1.0):
    B, L = p1.shape
    key, kprob, ku = jax.random.split(key, 3)
    do_cx = jax.random.bernoulli(kprob, pc, (B, 1))
    u = jax.random.uniform(ku, (B, L))
    beta = jnp.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta_c + 1.0)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
    )
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    c1 = jnp.where(do_cx, c1, p1)
    c2 = jnp.where(do_cx, c2, p2)
    return key, jnp.clip(c1, lower, upper), jnp.clip(c2, lower, upper)

def polynomial_mutation(key, X, pm=0.01, eta_m=20.0, lower=0.0, upper=1.0):
    N, L = X.shape
    key, kmask, kr = jax.random.split(key, 3)
    mask = jax.random.bernoulli(kmask, pm, X.shape)
    r = jax.random.uniform(kr, X.shape)
    xl = jnp.full((1, L), lower) if np.isscalar(lower) else jnp.array(lower)[None, :]
    xu = jnp.full((1, L), upper) if np.isscalar(upper) else jnp.array(upper)[None, :]
    width = xu - xl + 1e-12
    delta1 = (X - xl) / width
    delta2 = (xu - X) / width
    mut_pow = 1.0 / (eta_m + 1.0)
    r_less = (r < 0.5)
    xy1 = 1.0 - delta1
    val1 = 2.0 * r + (1.0 - 2.0 * r) * (xy1 ** (eta_m + 1.0))
    deltaq1 = (val1 ** mut_pow) - 1.0
    xy2 = 1.0 - delta2
    val2 = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy2 ** (eta_m + 1.0))
    deltaq2 = 1.0 - (val2 ** mut_pow)
    deltaq = jnp.where(r_less, deltaq1, deltaq2)
    X_mut = X + jnp.where(mask, deltaq * width, 0.0)
    return key, jnp.clip(X_mut, xl, xu)

def _tchebycheff(F, lamb, z):
    return jnp.max(lamb[None, :] * jnp.abs(F - z[None, :]), axis=-1)

# --- 工具：保存参数 ---
def flat_to_params_tree(flat_vector):
    flat1 = jnp.array([flat_vector])
    this_dict = policy.format_params_fn(flat1)
    new_dict = unfreeze(this_dict)
    for m_ in new_dict:
        for p_ in new_dict[m_]:
            for k_ in new_dict[m_][p_]:
                new_dict[m_][p_][k_] = new_dict[m_][p_][k_][0]
    return freeze(new_dict)

def save_best_params(params_tree):
    filename = f"{pde}_{METHOD_NAME}_{NET_ARCH}.msgpack"
    path = PARAMS_DIR / filename
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        f.write(serialization.to_bytes(params_tree))
    os.replace(tmp_path, path)
    return path

# --- MOEAD 主函数 ---
def moead(get_fitness, n=50, m=260, generations=2000, seed=0,
          lower=0.0, upper=1.0, pc=0.9, pm=None, eta_c=15.0, eta_m=20.0,
          T=None, delta=0.9, nr=2, verbose=True):

    if pm is None: pm = 1.0 / m
    if T is None:  T = max(5, int(0.1 * n))
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    pop = jax.random.uniform(k0, (n, m), minval=lower, maxval=upper)

    # --- 修复 1: 确保 objs 至少是二维 ---
    objs = get_fitness(pop)
    if objs.ndim == 1: 
        objs = objs[:, None]
    k = int(objs.shape[1])

    key, kw = jax.random.split(key)
    lambdas = jax.random.dirichlet(kw, jnp.ones((k,)), (n,))
    dists = jnp.linalg.norm(lambdas[:, None, :] - lambdas[None, :, :], axis=-1)
    neighbors = jnp.argsort(dists, axis=1)[:, :T]
    z = jnp.min(objs, axis=0)

    best_loss = float("inf")
    loss_ls, iter_time_ls = [], []

    for gen in range(1, generations + 1):
        t0 = time.time()
        key, kperm = jax.random.split(key)
        order = jax.random.permutation(kperm, n)

        for i in order.tolist():
            Bi = neighbors[i]
            key, kprob = jax.random.split(key)
            use_nb = jax.random.bernoulli(kprob, delta)
            if use_nb:
                key, ksel = jax.random.split(key)
                parents = jax.random.choice(ksel, Bi, shape=(2,), replace=False)
            else:
                key, ksel = jax.random.split(key)
                parents = jax.random.randint(ksel, (2,), 0, n)

            p1, p2 = pop[parents[0]][None, :], pop[parents[1]][None, :]
            key, c1, c2 = sbx_crossover(key, p1, p2, pc, eta_c, lower, upper)
            key, kpick = jax.random.split(key)
            child = jnp.where(jax.random.bernoulli(kpick, 0.5), c1, c2)
            key, child = polynomial_mutation(key, child, pm, eta_m, lower, upper)
            child = child[0]

            # --- 修复 2: 确保 child_obj 至少是一维 ---
            child_obj = get_fitness(child[None, :])[0]
            if np.ndim(child_obj) == 0:   
                child_obj = np.array([child_obj])

            z = jnp.minimum(z, child_obj)

            key, knei = jax.random.split(key)
            Bi_perm = jax.random.permutation(knei, Bi.shape[0])
            replaced = 0
            for jj in Bi[Bi_perm].tolist():
                f_old = _tchebycheff(objs[jj:jj+1, :], lambdas[jj], z)[0]
                f_new = _tchebycheff(child_obj[None, :], lambdas[jj], z)[0]
                if f_new <= f_old:
                    pop = pop.at[jj].set(child)
                    objs = objs.at[jj].set(child_obj)
                    replaced += 1
                    if replaced >= nr: break

        # --- 当前最好个体 ---
        sum_losses = jnp.sum(objs, axis=1)
        best_idx = int(jnp.argmin(sum_losses))
        best_losses = objs[best_idx]
        best_total = float(jnp.sum(best_losses))
        loss_ls.append(best_total)
        iter_time_ls.append(time.time() - t0)

        if best_total < best_loss:
            best_loss = best_total
            best_w = np.array(pop[best_idx])
            params_tree_best = flat_to_params_tree(best_w)
            save_path = save_best_params(params_tree_best)
            if verbose:
                print(f"💾 [iter {gen}] 新 best loss 保存：{best_loss:.4e} → {save_path}")

        if verbose:
            print(f"iter {gen}: best_total={best_total:.6f}, losses={best_losses.tolist()}")

    return best_w, best_loss, loss_ls, iter_time_ls


# --- 主程序 ---
if __name__ == '__main__':
    n, m = 50, policy.num_params
    generations = 10
    best_w, best_loss, loss_ls, iter_time_ls = moead(get_fitness, n=n, m=m, generations=generations)

    # --- 画 loss 曲线 ---
    fig_path = LOSS_DIR / f"{pde}_loss_iter.png"
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_ls) + 1), loss_ls, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'{pde} MOEAD Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"📈 曲线已保存：{fig_path}")

    # --- 结果 CSV ---
    flat_best = jnp.array([best_w])
    new_dict = unfreeze(policy.format_params_fn(flat_best))
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
        df = pd.DataFrame({"x": X_input[:,0], "t": X_input[:,1] if X_input.shape[1]>1 else 0,
                           "u_true": Y_true[:,0], "u_pred": u_pred[:,0]})
    elif Y_true.shape[1] == 2:
        u_pred, v_pred = np.asarray(derivs['u']), np.asarray(derivs['v'])
        df = pd.DataFrame({"x": X_input[:,0], "t": X_input[:,1] if X_input.shape[1]>1 else 0,
                           "u_true": Y_true[:,0], "v_true": Y_true[:,1],
                           "u_pred": u_pred[:,0], "v_pred": v_pred[:,0]})
    else:
        raise ValueError(f"Unsupported output dimension: {Y_true.shape[1]}")

    csv_path = RESULT_DIR / f"{pde}_Result.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ 数据已保存：{csv_path}")

    # --- 累计时间和 loss ---
    iter_time_cumsum = np.cumsum(iter_time_ls)
    df_log = pd.DataFrame({"iter": np.arange(1, len(loss_ls)+1),
                           "cum_time": iter_time_cumsum,
                           "loss": loss_ls})
    loss_time_csv_path = LOSS_TIME / f"{pde}_IterTime_Loss.csv"
    df_log.to_csv(loss_time_csv_path, index=False)
    print(f"✅ 累计耗时与损失已保存：{loss_time_csv_path}")

    final_param_file = PARAMS_DIR / f"{pde}_{METHOD_NAME}_{NET_ARCH}.msgpack"
    print(f"🎯 训练最优参数已存为：{final_param_file}")
