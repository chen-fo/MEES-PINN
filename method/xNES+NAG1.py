# XNES+NAG.py
import os, sys
PROJECT_ROOT = "/home/chenfanke/TaskPINN"
os.environ.pop("PYTHONPATH", None)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# --- imports ---
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from jax import random, numpy as jnp
from jax.scipy.linalg import expm
from jax import jit
from flax.core.frozen_dict import unfreeze, freeze
from pathlib import Path

# ===== 路径与输出目录 =====
PROJECT_ROOT = Path("/home/chenfanke/TaskPINN")
TARGET_BASE  = PROJECT_ROOT / "train" / "XNES+NAG1"
LOSS_DIR     = TARGET_BASE / "loss_iters"
RESULT_DIR   = TARGET_BASE / "result"
LOSS_TIME    = TARGET_BASE / "loss_time_csv"
LOSS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOSS_TIME.mkdir(parents=True, exist_ok=True)

# ===== 选择 PDE（按需修改以匹配你已有的模块接口）=====
pde = "Wave2D_LongTime"
from src.pde.Wave2D_LongTime import get_fitness, policy, train_task
# =================================
seed = 1

def xNES_NAG(
    f, x0, *,
    bs=100, lr=1e-2, sigma=1e-3,
    momentum=0.9, max_iters=10_000, verbose=True
):
    """xNES + NAG（近似式）：lr 控协方差 A；NAG 用动量 lookahead 近似（无 lr2）。"""
    key, rng = random.split(random.PRNGKey(seed))

    center = x0.copy()
    dim = int(center.shape[0])
    I = jnp.eye(dim)
    A = I * sigma

    # NAG 动量缓冲
    v = jnp.zeros(dim)

    bestFitness = -jnp.inf
    bestFound = None

    loss_ls = []
    iter_time_ls = []
    runtime = 0.0

    @jit
    def project_sample(center, A, v, rng):
        center_lookahead = center + momentum * v
        key, rng = random.split(rng)
        samples = random.normal(key, (bs, dim))
        samples_o = samples @ A + center_lookahead
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
    def update(center, A, v, utilities, samples):
        # 自然梯度中心方向
        grad_c = A @ (utilities @ samples)  # (dim,)

        # ---- NAG 近似更新（）----
        v = momentum * v + grad_c
        center = center + (momentum * v + grad_c)

        # ---- 协方差更新（全矩阵）----
        s_outer = samples[:, :, None] * samples[:, None, :]  # (bs, dim, dim)
        covGrad = jnp.sum(utilities[:, None, None] * (s_outer - I), axis=0)
        A = A @ expm(0.5 * lr * covGrad)
        return center, A, v

    numEvals = 0
    for it in range(max_iters):
        t0 = time.time()

        samples, samples_o, rng = project_sample(center, A, v, rng)
        fitnesses = f(samples_o)  # shape [bs]

        # 训练日志中的 loss 定义为 -平均适应度
        avg_fit = float(jnp.mean(fitnesses))
        loss_iter = -avg_fit
        loss_ls.append(loss_iter)

        # 跟踪当前最优个体
        cur_best = float(jnp.max(fitnesses))
        if cur_best > bestFitness:
            idx = int(jnp.argmax(fitnesses))
            bestFitness = cur_best
            bestFound = np.array(samples_o[idx])

        numEvals += bs

        if verbose:
            runtime += (time.time() - t0)
            print(f"iter={it+1:5d}  time={runtime:6.2f}s  loss={loss_iter:.2e}")

        utilities = compute_utilities(fitnesses)
        center, A, v = update(center, A, v, utilities, samples)

        iter_time_ls.append(float(time.time() - t0))

    print(f"\nFinished at iter={max_iters}, last loss={loss_ls[-1]:.2e}, best loss={min(loss_ls):.2e}")
    return bestFound, bestFitness, numEvals, iter_time_ls, loss_ls

# ===== 训练=====
max_iters = 10000
w0 = jnp.zeros(policy.num_params)
best_w, best_fit, evals, iter_time_ls, loss_ls = xNES_NAG(
    get_fitness, w0,
    bs=100, lr=1e-2, sigma=0.001, momentum=0.9,
    max_iters=max_iters, verbose=True
)

# ===== 保存 loss 曲线到 loss_iters/ =====
fig_path = LOSS_DIR / f"{pde}_loss_iter.png"
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_ls) + 1), loss_ls, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'{pde} Loss Curve (xNES+NAG)', fontsize=14)
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
    for p_ in new_dict[m]:
        for k in new_dict[m][p_]:
            new_dict[m][p_][k] = new_dict[m][p_][k][0]
params_tree = freeze(new_dict)

X_input = train_task.X_candidate
Y_true  = train_task.u_ref
model   = train_task.net
derivs  = model.derivatives(params_tree, X_input)

# ===== 导出结果到 result/ =====
if Y_true.shape[1] == 1:
    u_pred = np.asarray(derivs['u'])
    df = pd.DataFrame({
        'x': X_input[:, 0],
        'y': X_input[:, 1] if X_input.shape[1] >= 2 else 0,
        't': X_input[:, 2] if X_input.shape[1] == 3 else 0,
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

# ===== 累计耗时-损失 CSV 到 loss_time_csv/ =====
iter_time_cumsum = np.cumsum(iter_time_ls)
df_log = pd.DataFrame({
    "iter": np.arange(1, len(loss_ls) + 1, dtype=int),
    "cum_time": iter_time_cumsum,
    "loss": loss_ls
})
loss_time_csv_path = LOSS_TIME / f"{pde}_IterTime_Loss.csv"
df_log.to_csv(loss_time_csv_path, index=False)
print(f"✅ 累计耗时与损失已保存：{loss_time_csv_path}")
