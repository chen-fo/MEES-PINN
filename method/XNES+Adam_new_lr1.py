# XNES+Adam.py，新方法，包含lr2
import os, sys
PROJECT_ROOT = "/home/chenfanke/TaskPINN"
os.environ.pop("PYTHONPATH", None)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# --- imports ---
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from jax import random, numpy as jnp
from jax.scipy.linalg import expm
from jax import jit
from flax.core.frozen_dict import unfreeze, freeze
from flax import serialization
from pathlib import Path
from jax import config
config.update("jax_disable_jit", True) 

PROJECT_ROOT = Path("/home/chenfanke/TaskPINN")
TARGET_BASE  = PROJECT_ROOT / "train" / "XNES+Adam_new1"
LOSS_DIR     = TARGET_BASE / "loss_iters"
RESULT_DIR   = TARGET_BASE / "result"
LOSS_TIME    = TARGET_BASE / "loss_time_csv"
PARAMS_DIR   = TARGET_BASE / "params"
LOSS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOSS_TIME.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

# ===== 选择 PDE（只改这里）=====
pde = "Burgers1D"
from src.pde.Burgers1D import get_fitness, policy, train_task
# =================================

# ===== 命名配置（方程名_方法_网络参数）=====
method_name = "XNES+Adam_new1"  # 建议与目录一致
net_arch    = "4*8"            # 按你的网络结构填写

seed = 1

# 原始lr=1e-2
#调参lr=2e-2
def xNES_Adam(f, x0, *, bs=100, lr=2e-2, lr2=0.003, sigma=1e-3, max_iters=10_000,
              beta1=0.9, beta2=0.999, eps=1e-8, verbose=True):
    key, rng = random.split(random.PRNGKey(seed))

    center = x0.copy()
    dim = int(center.shape[0])
    I = jnp.eye(dim)
    A = I * sigma

    # Adam moments
    m = jnp.zeros(dim)
    v = jnp.zeros(dim)
    t_adam = 0

    bestFitness = -jnp.inf
    bestFound = None

    loss_ls = []
    iter_time_ls = []
    runtime = 0.0

    @jit
    def project_sample(center, A, rng):
        key, rng = random.split(rng)
        samples = random.normal(key, (bs, dim))
        samples_o = samples @ A + center
        return samples, samples_o, rng

    @jit
    def compute_utilities(fitnesses):
        order = jnp.argsort(fitnesses)                 # 升序索引
        ranks = jnp.argsort(order).astype(jnp.float32) # 0..bs-1
        L = fitnesses.size
        u_raw = jnp.log(L / 2.0 + 1.0) - jnp.log(L - ranks)
        utilities = jnp.maximum(0.0, u_raw)
        utilities = utilities / jnp.sum(utilities)
        utilities = utilities - 1.0 / L
        return utilities

    @jit
    def update(center, A, m, v, t_adam, utilities, samples):
        # 自然梯度中心方向
        grad_c = A @ (utilities @ samples)

        # Adam 更新（学习率用 lr2）
        m = beta1 * m + (1.0 - beta1) * grad_c
        v = beta2 * v + (1.0 - beta2) * (grad_c ** 2)
        t_adam += 1
        m_hat = m / (1.0 - beta1 ** t_adam)
        v_hat = v / (1.0 - beta2 ** t_adam)
        center = center + lr2 * m_hat / (jnp.sqrt(v_hat) + eps)

        # 协方差更新（全矩阵）
        covGrad = jnp.sum(
            utilities[:, None, None] *
            (samples[:, :, None] * samples[:, None, :] - I),
            axis=0
        )
        A = A @ expm(0.5 * lr * covGrad)
        return center, A, m, v, t_adam

    # --- 工具：展平向量 -> params_tree ---
    def flat_to_params_tree(flat_vector):
        flat1 = jnp.array([flat_vector])  # (1, P)
        this_dict = policy.format_params_fn(flat1)
        new_dict = unfreeze(this_dict)
        for m_ in new_dict:
            for p_ in new_dict[m_]:
                for k_ in new_dict[m_][p_]:
                    new_dict[m_][p_][k_] = new_dict[m_][p_][k_][0]  # 去 batch 维
        return freeze(new_dict)

    # --- 工具：保存最优参数（原子写入；覆盖式命名）---
    def save_best_params(params_tree):
        filename = f"{pde}_{method_name}_{net_arch}.msgpack"
        path = PARAMS_DIR / filename
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            f.write(serialization.to_bytes(params_tree))
        os.replace(tmp_path, path)
        return path

    numEvals = 0
    for it in range(max_iters):
        t0 = time.time()

        samples, samples_o, rng = project_sample(center, A, rng)
        fitnesses = f(samples_o)  # shape [bs]

        # 训练日志中的 loss 定义为 -平均适应度（与你参考范式一致）
        avg_fit = float(jnp.mean(fitnesses))
        loss_iter = -avg_fit
        loss_ls.append(loss_iter)

        # 跟踪最优个体（最大适应度），更优即刻保存
        cur_best = float(jnp.max(fitnesses))
        if cur_best > float(bestFitness):
            idx = int(jnp.argmax(fitnesses))
            bestFitness = cur_best
            bestFound = np.array(samples_o[idx])
            # 还原成 params_tree 并保存
            params_tree_best = flat_to_params_tree(bestFound)
            save_path = save_best_params(params_tree_best)
            if verbose:
                print(f"💾 [iter {it+1}] 新 best 保存：fitness={bestFitness:.4e} → {save_path}")

        numEvals += bs

        if verbose:
            runtime += (time.time() - t0)
            print(f"iter={it+1:5d}  time={runtime:6.2f}s  loss={loss_iter:.2e}")

        utilities = compute_utilities(fitnesses)
        center, A, m, v, t_adam = update(center, A, m, v, t_adam, utilities, samples)

        iter_time_ls.append(float(time.time() - t0))

    print(f"\nFinished at iter={max_iters}, last loss={loss_ls[-1]:.2e}, best loss={min(loss_ls):.2e}")
    return bestFound, bestFitness, numEvals, iter_time_ls, loss_ls

# ===== 训练（固定5,000 次；其余参数保留原风格）=====
max_iters = 5000
w0 = jnp.zeros(policy.num_params)
best_w, best_fit, _, iter_time_ls, loss_ls = xNES_Adam(
    get_fitness, w0,
    bs=100, lr=2e-2, lr2=0.003, sigma=0.001,
    max_iters=max_iters,
    beta1=0.9, beta2=0.999, eps=1e-8,
    verbose=True
)
# ===== 保存 loss 曲线到 loss_iters/ =====
fig_path = LOSS_DIR / f"{pde}_loss_iter.png"
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_ls) + 1), loss_ls, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'{pde} Loss Curve', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"📈 曲线已保存：{fig_path}")

# ===== 恢复最优参数 → 做预测 =====
flat_best = jnp.array([best_w])
this_dict = policy.format_params_fn(flat_best)
new_dict = unfreeze(this_dict)
for m in new_dict:
    for p in new_dict[m]:
        for k in new_dict[m][p]:
            new_dict[m][p][k] = new_dict[m][p][k][0]
params_tree = freeze(new_dict)

X_input = train_task.X_candidate
Y_true  = train_task.u_ref
model   = train_task.net
derivs  = model.derivatives(params_tree, X_input)

# ===== 结果 CSV 保存到 result/ =====
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
print(f"✅ 数据已保存：{csv_path}")

iter_time_cumsum = np.cumsum(iter_time_ls)
df_log = pd.DataFrame({
    "iter": np.arange(1, len(loss_ls) + 1, dtype=int),
    "cum_time": iter_time_cumsum,
    "loss": loss_ls
})
loss_time_csv_path = LOSS_TIME / f"{pde}_IterTime_Loss.csv"
df_log.to_csv(loss_time_csv_path, index=False)
print(f"✅ 累计耗时与损失已保存：{loss_time_csv_path}")

# ===== 结束提示：最优参数文件路径 =====
final_param_file = PARAMS_DIR / f"{pde}_{method_name}_{net_arch}.msgpack"
print(f"🎯 训练最优参数已存为：{final_param_file}")
