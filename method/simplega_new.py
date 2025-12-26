# ===== GPU & 路径环境 =====
import os, sys, time
from pathlib import Path

PROJECT_ROOT = "/home/chenfanke/TaskPINN"
os.environ.pop("PYTHONPATH", None)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# ===== 常用库 =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from jax import numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze
from flax import serialization          # 🔥 新增
from evojax.algo import SimpleGA

# ===== 统一保存目录 =====
PROJECT_ROOT = Path("/home/chenfanke/TaskPINN")
TARGET_BASE  = PROJECT_ROOT / "train" / "simpleGa"
LOSS_DIR     = TARGET_BASE / "loss_iters"
RESULT_DIR   = TARGET_BASE / "result"
LOSS_TIME    = TARGET_BASE / "loss_time_csv"
PARAMS_DIR   = TARGET_BASE / "params"     # 🔥 新增
for d in [LOSS_DIR, RESULT_DIR, LOSS_TIME, PARAMS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===== 配置 =====
seed        = 1
pop_size    = 50
max_iters   = 5000  
max_time    = 60        

# ===== PDE =====
pde = "Wave2D_LongTime"
from src.pde.Wave2D_LongTime import get_fitness, policy, train_task

# ===== 方法命名，用于存文件 =====
method_name = "SimpleGA"
net_arch    = "4*8"   # 你可以按网络结构写，比如 "4*8"

# ===== 初始化优化器 =====
solver = SimpleGA(
    pop_size=pop_size,
    param_size=policy.num_params,
    seed=seed,
)

# ===== 工具函数 =====
def flat_to_params_tree(flat_vector):
    flat1 = jnp.array([flat_vector])  # (1, P)
    this_dict = policy.format_params_fn(flat1)
    new_dict = unfreeze(this_dict)
    for m in new_dict:
        for p in new_dict[m]:
            for k in new_dict[m][p]:
                new_dict[m][p][k] = new_dict[m][p][k][0]  # 去 batch 维
    return freeze(new_dict)

def save_best_params(params_tree, iter_idx, loss_val):
    filename = f"{pde}_{method_name}_{net_arch}.msgpack"
    path = PARAMS_DIR / filename
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        f.write(serialization.to_bytes(params_tree))
    os.replace(tmp_path, path)
    print(f"💾 [iter {iter_idx}] 新 best 保存：loss={loss_val:.4e} → {path}")
    return path

# ===== 训练循环 =====
loss_ls = []
iter_time_ls = []
runtime = 0.0
train_iters = 0
best_loss = float("inf")
best_params_tree = None

while train_iters < max_iters:
    t0 = time.time()

    params = solver.ask()
    scores = get_fitness(params)           # 越大越好
    solver.tell(fitness=scores)

    avg_loss = np.mean(np.array(scores, copy=False))
    loss_iter = -avg_loss
    loss_ls.append(loss_iter)

    elapsed = time.time() - t0
    iter_time_ls.append(elapsed)
    runtime += elapsed
    train_iters += 1

    #  保存最优
    if loss_iter < best_loss:
        best_loss = loss_iter
        idx_best = int(np.argmin(np.array(scores)))
        best_params = params[idx_best]
        best_params_tree = flat_to_params_tree(best_params)
        save_best_params(best_params_tree, train_iters, best_loss)

    print(f"iter={train_iters:5d}  time={runtime:6.2f}s  loss={loss_iter:.2e}")

print(f"\nFinished at iter={train_iters}, last loss={loss_ls[-1]:.2e}, best loss={min(loss_ls):.2e}")

# ===== 保存 loss 曲线 =====
fig_path = LOSS_DIR / f"{pde}_loss_iter.png"
plt.figure(figsize=(10, 6))
plt.plot(range(1, train_iters + 1), loss_ls, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'{pde} Loss Curve (GA)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"📈 曲线已保存：{fig_path}")

# ===== 模型预测与结果保存（基于最优参数）=====
if best_params_tree is None:
    raise RuntimeError("没有保存到任何最优参数！")

X_input = train_task.X_candidate
Y_true  = train_task.u_ref
model   = train_task.net
derivs  = model.derivatives(best_params_tree, X_input)

def _xyt(X):
    X = np.asarray(X)
    x = X[:, 0]
    y = X[:, 1] if X.shape[1] >= 2 else 0
    t = X[:, 2] if X.shape[1] >= 3 else 0
    return x, y, t

x, y, t = _xyt(X_input)

if Y_true.shape[1] == 1:
    u_pred = np.asarray(derivs['u'])
    df = pd.DataFrame({'x': x, 'y': y, 't': t,
                       'u_true': Y_true[:, 0], 'u_pred': u_pred[:, 0]})
elif Y_true.shape[1] == 2:
    u_pred = np.asarray(derivs['u'])
    v_pred = np.asarray(derivs['v'])
    df = pd.DataFrame({'x': x, 'y': y, 't': t,
                       'u_true': Y_true[:, 0], 'v_true': Y_true[:, 1],
                       'u_pred': u_pred[:, 0], 'v_pred': v_pred[:, 0]})
elif Y_true.shape[1] == 3:
    u_pred = np.asarray(derivs['u'])
    v_pred = np.asarray(derivs['v'])
    p_pred = np.asarray(derivs['p'])
    df = pd.DataFrame({'x': x, 'y': y, 't': t,
                       'u_true': Y_true[:, 0], 'v_true': Y_true[:, 1], 'p_true': Y_true[:, 2],
                       'u_pred': u_pred[:, 0], 'v_pred': v_pred[:, 0], 'p_pred': p_pred[:, 0]})
else:
    raise ValueError(f"Unsupported output dimension: {Y_true.shape[1]}")

csv_path = RESULT_DIR / f"{pde}_Result.csv"
df.to_csv(csv_path, index=False)
print(f"✅ 数据已保存：{csv_path}")

# ===== 迭代耗时-损失 CSV =====
iter_time_cumsum = np.cumsum(iter_time_ls)
df_log = pd.DataFrame({
    "iter": np.arange(1, len(loss_ls) + 1, dtype=int),
    "cum_time": iter_time_cumsum,
    "loss": loss_ls
})
loss_time_csv_path = LOSS_TIME / f"{pde}_IterTime_Loss.csv"
df_log.to_csv(loss_time_csv_path, index=False)
print(f"✅ 累计耗时与损失已保存：{loss_time_csv_path}")

# ===== 最终提示 =====
final_param_file = PARAMS_DIR / f"{pde}_{method_name}_{net_arch}.msgpack"
print(f"🎯 训练最优参数已存为：{final_param_file}")
