# -*- coding: utf-8 -*-
import jax
from jax import random
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
from src.data import DataSampler_T, LowDiscrepancySampler
from typing import Sequence, Tuple
import numpy as np

from EAPINN import geometry
from src.utils import SimManager, stack_outputs
from src.nn import BaseNN
from pathlib import Path

# -------------------- 路径（保持与 KdV 一致） --------------------
project_root = Path(__file__).resolve().parent.parent.parent
ref_dir = project_root / 'ref'

# -------------------- Batch 配置（保持与 KdV 一致） --------------------
BatchSize_eq = 8192
BatchSize_data = 4096

# -------------------- State --------------------
@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray
    bcs_masks: Sequence[jnp.ndarray]  # 兼容 KdV 的接口（本 PDE 不使用 bc objects）

# -------------------- PINN：导数计算（只对 x 求导，输入可为 (x,t) 或 (x,)） --------------------
class PINN(BaseNN):
    """PINN for 1D steady convection-diffusion: v*u_x = k*u_xx
       Network itself receives x (1-dim) as input; derivatives() accepts X which can be (N,2) (x,t)
       or (N,1) (x,), and computes u, u_x, u_xx.
    """
    def derivatives(self, params, X):
        """
        输入:
            X: (N,2) 或 (N,1) 其中列顺序为 (x, t) 或 (x,)
        返回:
            dict{'u','u_x','u_xx'} -> 每项形状 (N,1)
        """
        # 取 x 列：假定第一列是 x (与 KdV 的约定一致)
        x_col = X[:, 0:1]  # (N,1)

        def forward(z):
            # 网络输入是一维（只喂 x）
            out = self.apply(params, z[None, :])   # (1, out_dim)
            return out[0]                          # (out_dim,)

        def u_fn(z):
            # z 形如 (1,) -> 返回标量 u(x)
            return forward(z)[0]

        # 计算一阶、二阶导数（针对 x）
        grads_u = jax.vmap(jax.grad(u_fn))(x_col)        # (N,1) -> grad wrt x
        hess_u  = jax.vmap(jax.hessian(u_fn))(x_col)     # (N,1,1)

        u = jax.vmap(u_fn)(x_col).reshape(-1, 1)

        return {
            'u':    u,
            'u_x':  grads_u[:, 0:1],
            'u_xx': hess_u[:, 0, 0].reshape(-1, 1),
        }

# -------------------- 任务 --------------------
class Diffusion1D(VectorizedTask):
    """
    稳态一维对流–扩散方程（embedded in time-domain for sampler compatibility）
        v * u_x = k * u_xx,  x ∈ [x_l, x_u], t ∈ [t0, t1] (t is dummy)
    边界条件： u(0,t)=0, u(L,t)=1 (软约束, 使用解析解)
    """
    def __init__(self, datapath=ref_dir / 'diffusion.csv', bbox=[0.0, 1.0, 0.0, 1.0], n_grid: int = 4096):
        # --- 基础设置 ---
        self.max_steps = 1
        # 任务对外呈现的是 (x,t) 两维采样（兼容 DataSampler_T/KdV 约定）
        self.obs_shape = tuple([2, ])   # (x, t) 列顺序与 KdV 一致 -> X[:,0]=x, X[:,1]=t
        self.act_shape = tuple([1, ])   # 占位（policy 返回的是衍生量堆叠）

        # --- PDE 参数（放类内） ---
        self.v = 6.0
        self.k = 1.0
        self.L = 1.0
        self.x_l, self.x_u = float(bbox[0]), float(bbox[1])
        # 时间区间 (只是为了采样)
        self.t0, self.t1 = float(bbox[2]), float(bbox[3])

        # --- 域定义（空间 Interval + 时间 TimeDomain -> GeometryXTime） ---
        # 注意：Geometry/TimeDomain 接口使用位置参数（与你项目中的 KdV 写法对齐）
        space = geometry.Interval(self.x_l, self.x_u)
        time_domain = geometry.TimeDomain(self.t0, self.t1)
        self.geom = space
        self.geom_time = geometry.GeometryXTime(self.geom, time_domain)

        # --- 网络定义（网络实际只接受 x） ---
        self.output_dim = 1
        self.input_dim = 1      # 网络输入是 x（1D）
        self.net = PINN(input_dim=self.input_dim, output_dim=self.output_dim)

        # 参数初始化
        self.seed = 0
        self._init_params()
        self.format_params_fn = jax.vmap(self.fmt)   # 向量化 format 函数
        self.num_params = self.param_size

        # 告诉 policy 需要哪些导数
        self.layout = ['u', 'u_x', 'u_xx']

        # 损失权重
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.data_lambda = 0.0  # 默认不使用外部监督数据

        # 解析解（用于 BC）
        self.Pe = self.v * self.L / self.k
        def eval_u(x):
            return (1.0 - jnp.exp(self.Pe * x / self.L)) / (1.0 - jnp.exp(self.Pe))
        self.eval_u = eval_u

        # --- 构造备用监督数据（形状与采样器输出一致：两列 (x,t)） ---
        x_all = jnp.linspace(self.x_l, self.x_u, n_grid).reshape(-1, 1)      # (N,1)
        t_zeros = jnp.zeros_like(x_all)                                     # (N,1) t=0
        X_xt = jnp.hstack([x_all, t_zeros])   # (N,2) 列为 (x,t)
        y_all = self.eval_u(x_all).reshape(-1, 1)
        self.X_data = X_xt
        self.Y_data = y_all
        self.X_candidate = X_xt
        self.u_ref = y_all

        # --- PDE 采样器（传入 geom_time 和 ic_bcs 列表） ---
        # 稳态问题没有 IC，这里传空列表让 DataSampler_T 接受参数
        self.bcs = []
        self.pde_data = DataSampler_T(self.geom_time, self.bcs, mul=4).train_x_all  # (N_eq, 2) -> (x,t)
        self.X_pde = self.pde_data
        self.Y_pde = np.zeros(shape=(len(self.X_pde), self.output_dim), dtype=np.float32)

        # --- mini-batch & 采样器 ---
        self.BatchSize_eq = BatchSize_eq
        self.BatchSize_data = BatchSize_data
        # domain_bounds 顺序与 (x,t) 对齐
        domain_bounds = [
            [self.x_l, self.x_u],  # x
            [self.t0, self.t1],    # t
        ]
        self.pde_sampler = LowDiscrepancySampler(self.X_pde, self.Y_pde, domain_bounds)

        if len(self.X_data) > self.BatchSize_data:
            self.is_batch = True
            self.data_size = self.BatchSize_data
            self.data_sampler = LowDiscrepancySampler(self.X_data, self.Y_data, domain_bounds)
        else:
            self.is_batch = False
            self.data_size = len(self.X_data)

        # BC 相关占位（本实现直接在 loss 中用解析解约束端点）
        self.bcs_masks = []
        self.bcs_points = []

        # --- reset / step（与 KdV 相同风格） ---
        def reset_fn(key):
            X_eq, Y_eq = self.pde_sampler.get_batch(batch_size=self.BatchSize_eq)  # (Neq,2)
            if self.is_batch:
                X_d, Y_d = self.data_sampler.get_batch(batch_size=self.BatchSize_data)
            else:
                X_d, Y_d = self.X_data, self.Y_data
            masks = [bc.filter(X_eq) for bc in self.bcs] if len(self.bcs) > 0 else []
            X_batch = jnp.concatenate((X_eq, X_d), axis=0)  # (Neq+Ndata, 2)
            Y_batch = jnp.concatenate((Y_eq, Y_d), axis=0)  # (Neq+Ndata, 1)
            return State(obs=X_batch, labels=Y_batch, bcs_masks=masks)

        def step_fn(states, actions):
            def single_loss_fn(s, a):
                reward = -self.loss_fn(pred=a, X_batch=s.obs, Y_batch=s.labels)
                return reward

            rewards = jax.vmap(single_loss_fn)(states, actions)
            done = jnp.ones((actions.shape[0],), dtype=jnp.int32)
            return states, rewards, done

        self._reset_fn = reset_fn
        self._step_fn = step_fn

    # ---------- 参数初始化 / seed ----------
    def _init_params(self):
        key = random.PRNGKey(self.seed)
        dummy = jnp.zeros((1, self.input_dim))  # 网络只喂 x (1,)
        self.params_tree = self.net.init(key, dummy)
        self.param_size, self.fmt = get_params_format_fn(self.params_tree)

    def update_seed(self, seed):
        self.seed = seed
        self._init_params()

    # ---------- PDE 残差（基于 policy 输出的 pred）----------
    def pde_fn(self, pred):
        """
        r = v * u_x - k * u_xx
        layout: ['u', 'u_x', 'u_xx']
        pred: (N,3) -> columns [u, u_x, u_xx]
        """
        u_x  = pred[:, 1:2]
        u_xx = pred[:, 2:3]
        r_u = self.v * u_x - self.k * u_xx
        return r_u

    # ---------- 数据损失接口（保留） ----------
    def data_fn(self, Y_ref, pred):
        if self.data_lambda <= 0.0 or self.data_size == 0:
            return 0.0
        u_true = Y_ref[-self.data_size:, 0:1]
        u_pred = pred[-self.data_size:, 0:1]
        loss_tmp = (u_pred - u_true) ** 2
        data_loss = jnp.sum(loss_tmp) / pred.shape[0]
        return data_loss

    # ---------- 总损失：PDE(内部点) + BC(端点) (+ 可选 data) ----------
    def loss_fn(self, pred, X_batch, Y_batch):
        # 仅对 PDE 段计算残差（前 BatchSize_eq 个为 PDE 采样）
        pde_size = self.BatchSize_eq
        pde_pred = pred[:pde_size, :]    # (Neq, 3)
        X_pde    = X_batch[:pde_size, :] # (Neq, 2) -> (x,t)

        # 提取 x（假定第一列是 x）
        x = X_pde[:, 0:1]

        # PDE loss（排除边界点）
        r_all = self.pde_fn(pde_pred)    # (Neq,1)
        is_boundary = jnp.isclose(x, self.x_l) | jnp.isclose(x, self.x_u)
        interior_mask = (~is_boundary).astype(r_all.dtype)
        r_masked = r_all * interior_mask
        denom = jnp.maximum(jnp.sum(interior_mask), 1.0)
        pde_loss = jnp.sum(r_masked ** 2) / denom

        # BC loss（端点约束，使用解析解）
        u_true_boundary = self.eval_u(x)
        u_pred_boundary = pde_pred[:, 0:1]
        bc_err = (u_pred_boundary - u_true_boundary) * is_boundary.astype(u_pred_boundary.dtype)
        bc_denom = jnp.maximum(jnp.sum(is_boundary.astype(jnp.float32)), 1.0)
        bc_loss = jnp.sum(bc_err ** 2) / bc_denom

        # Data loss（默认 0）
        data_loss = self.data_fn(Y_batch, pred) if self.data_lambda > 0.0 else 0.0

        total_loss = (
            self.pde_lambda * pde_loss +
            self.bc_lambda * bc_loss +
            self.data_lambda * data_loss
        )

        jax.debug.print("pde_loss={:}, bc_loss={:}, data_loss={:}", pde_loss, bc_loss, data_loss)
        return total_loss

    # ---------- VectorizedTask 接口 ----------
    def reset(self, key):
        return self._reset_fn(key)

    def step(self, state, action):
        return self._step_fn(state, action)

# -------------------- Policy --------------------
class PINNsPolicy(PolicyNetwork):
    def __init__(self, net, num_params, format_params_fn, grad_keys):
        self.net = net
        self.num_params = num_params
        self.format_params_fn = format_params_fn
        self.grad_keys = grad_keys

    def get_actions(self,
                    t_states: TaskState,
                    flat_params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        # flat -> pytree (batched)
        params_tree = self.format_params_fn(flat_params)

        # obs 与 params 对齐；obs 形状 (B, N_pts, 2) -> 每个样本 obs_i 为 (N_pts,2) (x,t)
        obs = t_states.obs

        # forward + derivatives: 每个 params 对应一个 obs_i
        def f_single(params, obs_i):
            outs = self.net.derivatives(params, obs_i)  # dict with keys 'u','u_x','u_xx'
            return stack_outputs(outs, self.grad_keys)  # (N_pts, 3)

        actions = jax.vmap(f_single)(params_tree, obs)  # (B, N_pts, 3)
        return actions, p_states

# -------------------- Initialize task & policy --------------------
seeds = 0
train_task = Diffusion1D()
policy = PINNsPolicy(train_task.net,
                     train_task.num_params,
                     train_task.format_params_fn,
                     train_task.layout)

sim_mgr = SimManager(n_repeats=1, test_n_repeats=1, pop_size=0, n_evaluations=1,
                     policy_net=policy, train_vec_task=train_task, valid_vec_task=train_task,
                     seed=seeds)

def get_fitness(samples):
    scores, _ = sim_mgr.eval_params(params=samples, test=False)
    return scores
