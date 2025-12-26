import os, sys, time
PROJECT_ROOT = "/home/chenfanke/TaskPINN"
os.environ.pop("PYTHONPATH", None)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from jax import numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze
from evojax.algo import CMA_ES_JAX
from flax import serialization

# ===== 统一保存目录 =====
PROJECT_ROOT = Path("/home/chenfanke/TaskPINN")
TARGET_BASE  = PROJECT_ROOT / "train" / "CMAES"
LOSS_DIR     = TARGET_BASE / "loss_iters"
RESULT_DIR   = TARGET_BASE / "result"
LOSS_TIME    = TARGET_BASE / "loss_time_csv"
PARAMS_DIR   = TARGET_BASE / "params"
LOSS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOSS_TIME.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

# ===== 命名配置（方程名_方法_网络参数）=====
# 方法名建议与你的实验目录保持一致，便于查找
method_name = "CMAES"
# 网络结构
net_arch    = "4*8"

# === 配置 ===
seed        = 1
pop_size    = 50
init_stdev = 0.01
#init_stdev  = 0.02
elite_ratio = 0.5
w_decay     = 0.0
max_iters   = 5000
max_time    = 60

# === PDE（只改这里）===
pde = "convection_diffusion"
from src.pde.convection_diffusion import get_fitness, policy, train_task

# === 初始化优化器 ===
solver = CMA_ES_JAX(
    pop_size=pop_size,
    init_stdev=init_stdev,
    param_size=policy.num_params,
    seed=seed,
)

loss_ls = []
iter_time_ls = []
runtime = 0.0
train_iters = 0

# 维护全局 best（以“最小 loss”为准；注意 fitness = -loss）
best_loss = np.inf
best_flat_params = None  # 记录历史最优个体（展平向量）


# --- 把展平参数向量还原成 params_tree 的小工具 ---
def flat_to_params_tree(flat_vector):
    # flat_vector: shape (param_size,)
    flat1 = jnp.array([flat_vector])  # (1, P)
    this_dict = policy.format_params_fn(flat1)
    new_dict = unfreeze(this_dict)
    for m in new_dict:
        for p_ in new_dict[m]:
            for k in new_dict[m][p_]:
                new_dict[m][p_][k] = new_dict[m][p_][k][0]
    return freeze(new_dict)

# --- 保存最优参数（覆盖命名：方程名_方法_网络参数.msgpack）---
def save_best_params(params_tree, save_dir, pde_name, method_name, net_arch):
    filename = f"{pde_name}_{method_name}_{net_arch}.msgpack"
    path = save_dir / filename
    # 原子写：先写临时文件，再替换
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        f.write(serialization.to_bytes(params_tree))
    os.replace(tmp_path, path)
    return path

# === 训练循环 ===
while train_iters < max_iters:
    t0 = time.time()

    # 采样一批个体（展平参数）
    params = solver.ask()                # shape: (pop_size, param_size)
    scores = get_fitness(params)         # “越大越好”，通常 scores = -loss
    solver.tell(fitness=scores)

    # 统计用：这里沿用你的“avg_loss”记录
    avg_loss = np.mean(np.array(scores, copy=False))
    loss_ls.append(-avg_loss)

    # —— 用“当前批次最优个体”的 loss 与全局 best 比较，若更优则立刻保存 ——
    # 当前批次里分数最高（fitness 最大）即 loss 最小
    idx_best = int(np.argmax(scores))
    cur_best_loss = float(-scores[idx_best])  # fitness = -loss
    if cur_best_loss < best_loss:
        best_loss = cur_best_loss
        best_flat_params = np.array(params[idx_best], copy=True)
        # 还原成 tree 并保存
        params_tree = flat_to_params_tree(best_flat_params)
        save_path = save_best_params(params_tree, PARAMS_DIR, pde, method_name, net_arch)
        print(f"💾 [iter {train_iters+1}] 新 best 保存：loss={best_loss:.4e} → {save_path}")

    elapsed = time.time() - t0
    iter_time_ls.append(elapsed)
    runtime += elapsed
    train_iters += 1

    print(f"iter={train_iters:5d}  time={runtime:6.2f}s  loss(avg)={loss_ls[-1]:.2e}  best={best_loss:.2e}")

print(f"\nFinished at iter={train_iters}, last loss(avg)={loss_ls[-1]:.2e}, best loss={best_loss:.2e}")

# ===== 保存 loss 曲线到 loss_iters/ =====
fig_path = LOSS_DIR / f"{pde}_loss_iter.png"
plt.figure(figsize=(10, 6))
plt.plot(range(1, train_iters + 1), loss_ls, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss (avg over pop)', fontsize=12)
plt.title(f'{pde} Loss Curve', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"📈 曲线已保存：{fig_path}")

# === 用“历史最优个体”恢复参数树（用于预测导出）===
if best_flat_params is None:
    # 理论上不会发生；保险起见
    best_flat_params = np.array(solver.best_params, copy=True)
params_tree = flat_to_params_tree(best_flat_params)

# === 模型预测 ===
X_input = train_task.X_candidate
Y_true  = train_task.u_ref
model   = train_task.net
derivs  = model.derivatives(params_tree, X_input)

# === 兼容 1D/2D/(+t) 的列访问 ===
def _xyt(X):
    X = np.asarray(X)
    x = X[:, 0]
    y = X[:, 1] if X.shape[1] >= 2 else 0
    t = X[:, 2] if X.shape[1] >= 3 else 0
    return x, y, t

x, y, t = _xyt(X_input)

# ===== 结果 CSV 保存到 result/ =====
if Y_true.shape[1] == 1:
    u_pred = np.asarray(derivs['u'])
    df = pd.DataFrame({
        'x': x, 'y': y, 't': t,
        'u_true': Y_true[:, 0],
        'u_pred': u_pred[:, 0],
    })
elif Y_true.shape[1] == 2:
    u_pred = np.asarray(derivs['u'])
    v_pred = np.asarray(derivs['v'])
    df = pd.DataFrame({
        'x': x, 'y': y, 't': t,
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
        'x': x, 'y': y, 't': t,
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

# ===== 结束提示：当前最优参数文件路径 =====
final_param_file = PARAMS_DIR / f"{pde}_{method_name}_{net_arch}.msgpack"
print(f"🎯 训练最优参数已存为：{final_param_file}")
