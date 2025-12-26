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

import jax
import jax.numpy as jnp
from jax import random
from flax.core.frozen_dict import unfreeze, freeze
from flax import serialization

# ===== 统一保存目录 =====
PROJECT_ROOT = Path("/home/chenfanke/TaskPINN")
TARGET_BASE  = PROJECT_ROOT / "train" / "PSO"
LOSS_DIR     = TARGET_BASE / "loss_iters"
RESULT_DIR   = TARGET_BASE / "result"
LOSS_TIME    = TARGET_BASE / "loss_time_csv"
PARAMS_DIR   = TARGET_BASE / "params"
for d in [LOSS_DIR, RESULT_DIR, LOSS_TIME, PARAMS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===== 配置 =====
seed        = 1
pop_size    = 50
max_iters   = 5000   
max_time    = 60        
w           = 0.3
c1          = 1.5
c2          = 1.5
v_max       = 0.5
bound_down  = -2.0
bound_up    =  2.0

# ===== PDE =====
pde = "Wave2D_LongTime"
from src.pde.Wave2D_LongTime import get_fitness, policy, train_task

# 方法命名
method_name = "PSO"
net_arch    = "4*8"   # 按你的网络结构填写

# ===== PSO（适配 ask/tell/best_params 接口）=====
class PSOAdapter:
    def __init__(self, f, dim, pop_size, seed,
                 w=0.3, c1=1.5, c2=1.5, v_max=0.5,
                 low=-2.0, high=2.0):
        self.f = f
        self.dim = dim
        self.pop_size = pop_size
        self.w, self.c1, self.c2 = w, c1, c2
        self.v_max = v_max
        self.low, self.high = low, high

        self._key = random.PRNGKey(seed)
        self._key, k1, k2 = random.split(self._key, 3)
        # 初始化
        self.p = random.uniform(k1, shape=(pop_size, dim), minval=low, maxval=high)
        self.v = random.normal(k2, shape=(pop_size, dim)) * 0.5
        self.pb = self.p.copy()

        self.pbs = self.f(self.pb)  # 分数越大越好
        best_idx = jnp.argmax(self.pbs)
        self.gb = self.pb[best_idx]
        self.gbs = self.pbs[best_idx]

    def ask(self):
        return self.p

    def tell(self, fitness):
        better = fitness > self.pbs
        self.pb = jnp.where(better[:, None], self.p, self.pb)
        self.pbs = jnp.where(better, fitness, self.pbs)

        best_idx = jnp.argmax(self.pbs)
        self.gb = self.pb[best_idx]
        self.gbs = self.pbs[best_idx]

        self._key, k1, k2 = random.split(self._key, 3)
        r1 = random.normal(k1, shape=self.p.shape)
        r2 = random.normal(k2, shape=self.p.shape)

        self.v = (self.w * self.v
                  + self.c1 * r1 * (self.pb - self.p)
                  + self.c2 * r2 * (self.gb - self.p))
        self.v = jnp.clip(self.v, -self.v_max, self.v_max)
        self.p = self.p + self.v
        self.p = jnp.clip(self.p, self.low, self.high)

    @property
    def best_params(self):
        return np.array(self.gb)

# ===== 初始化优化器 =====
solver = PSOAdapter(
    f=get_fitness,
    dim=policy.num_params,
    pop_size=pop_size,
    seed=seed,
    w=w, c1=c1, c2=c2, v_max=v_max,
    low=bound_down, high=bound_up
)

# --- 工具函数：展平向量 -> params_tree ---
def flat_to_params_tree(flat_vector):
    flat1 = jnp.array([flat_vector])  # (1, P)
    this_dict = policy.format_params_fn(flat1)
    new_dict = unfreeze(this_dict)
    for m_ in new_dict:
        for p_ in new_dict[m_]:
            for k_ in new_dict[m_][p_]:
                new_dict[m_][p_][k_] = new_dict[m_][p_][k_][0]
    return freeze(new_dict)

# --- 工具函数：保存最优参数 ---
def save_best_params(params_tree):
    filename = f"{pde}_{method_name}_{net_arch}.msgpack"
    path = PARAMS_DIR / filename
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        f.write(serialization.to_bytes(params_tree))
    os.replace(tmp_path, path)
    return path

# ===== 训练循环 =====
loss_ls = []
iter_time_ls = []
runtime = 0.0
train_iters = 0
best_loss = float("inf")

while train_iters < max_iters:
    t0 = time.time()
    params = solver.ask()
    scores = get_fitness(params)
    solver.tell(fitness=scores)

    avg_score = np.mean(np.array(scores, copy=False))
    loss_iter = -avg_score
    loss_ls.append(loss_iter)

    elapsed = time.time() - t0
    iter_time_ls.append(elapsed)
    runtime += elapsed
    train_iters += 1

    # ---- 检查并保存最优 ----
    if loss_iter < best_loss:
        best_loss = loss_iter
        best_params_flat = solver.best_params
        params_tree_best = flat_to_params_tree(best_params_flat)
        save_path = save_best_params(params_tree_best)
        print(f"💾 [iter {train_iters}] 新 best 保存：loss={best_loss:.4e} → {save_path}")

    print(f"iter={train_iters:5d}  time={runtime:6.2f}s  loss={loss_iter:.2e}")

print(f"\nFinished at iter={train_iters}, last loss={loss_ls[-1]:.2e}, best loss={min(loss_ls):.2e}")

# ===== 保存 loss 曲线 =====
fig_path = LOSS_DIR / f"{pde}_loss_iter.png"
plt.figure(figsize=(10, 6))
plt.plot(range(1, train_iters + 1), loss_ls, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'{pde} Loss Curve (PSO)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"📈 曲线已保存：{fig_path}")

# ===== 恢复最优参数并预测 =====
flat_best = jnp.array([solver.best_params])
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
