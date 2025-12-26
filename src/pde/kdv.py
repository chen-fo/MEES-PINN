import jax
from jax import random
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
from src.data import DataSampler_T,LowDiscrepancySampler 
from typing import Sequence
import pandas as pd
import numpy as np

from EAPINN import geometry
from EAPINN.ICBC import IC
from src.utils import DataLoader, SimManager, addbc, stack_outputs
from src.nn import BaseNN
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent 
ref_dir = project_root / 'ref'

BatchSize_eq = 8192
BatchSize_data = 4096
@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray
    bcs_masks: Sequence[jnp.ndarray]

class PINN(BaseNN):
    """PINN for 1D KdV: u_t + v1*u*u_x + v2*u_xxx = 0"""
    def derivatives(self, params, X):
        """
        输入:
            X: (N, 2) 其中 X[:,0]=x, X[:,1]=t
        返回:
            dict {
                'u': (N,1),
                'u_x': (N,1),
                'u_xx': (N,1),
                'u_xxx': (N,1),
                'u_t': (N,1),
            }
        """
        # 单点前向
        def forward(z):
            out = self.apply(params, z[None, :])   # (1, out_dim)
            return out[0]                          # (out_dim,)

        # 标准模板：u_fn 接受 z 向量；标量输出 u
        def u_fn(z):
            return forward(z)[0]                   # 标量 u(x,t)

        # 一阶/二阶：按模板做 vmap(grad/hessian)(X)
        grads_u = jax.vmap(jax.grad(u_fn))(X)         # (N, 2) -> [∂u/∂x, ∂u/∂t]
        hess_u  = jax.vmap(jax.hessian(u_fn))(X)      # (N, 2, 2)

        # 三阶（仅 x 方向）：vmap(jacfwd^3)(X) -> (N, 2, 2, 2)
        third_u = jax.vmap(jax.jacfwd(jax.jacfwd(jax.jacfwd(u_fn))))(X)

        # 标量值
        u = jax.vmap(u_fn)(X).reshape(-1, 1)

        # 组装（命名与模板一致；KDV 需要的 x 向导数齐全）
        return {
            'u':    u,
            'u_x':  grads_u[:, 0:1],                 # ∂u/∂x
            'u_xx': hess_u[:, 0, 0].reshape(-1, 1),  # ∂²u/∂x²
            'u_xxx': third_u[:, 0, 0, 0].reshape(-1, 1),  # ∂³u/∂x³
            'u_t':  grads_u[:, 1:2],                 # ∂u/∂t
        }

class KdV1D(VectorizedTask):
    def __init__(self,datapath = ref_dir / 'kdv.csv',bbox = [0.0, 1.5, 0.0, 2.0]):
        # --- 基础设置 ---
        self.max_steps = 1
        self.obs_shape = tuple([2, ])   # (x, t)
        self.act_shape = tuple([1, ])   # 动作维度写法与模板一致

        V1 = 1.0
        V2 = 1e-3
        C1, C2 = 0.3, 0.1
        X1, X2 = 0.4, 0.8
        # 域定义（x 维 + t 维）
        self.bbox = bbox

        # 1D 空间：使用 Interval；时间域 TimeDomain
        space = geometry.Interval(bbox[0],bbox[1])
        time_domain = geometry.TimeDomain(bbox[2], bbox[3])
        self.geom = space
        self.geom_time = geometry.GeometryXTime(self.geom, time_domain)

        # --- 网络定义 ---
        self.output_dim = 1
        self.input_dim = self.geom_time.dim if self.geom_time is not None else 2
        self.net = PINN(input_dim=self.input_dim, output_dim=self.output_dim)

        # 参数初始化
        self.seed = 0
        self._init_params()
        self.format_params_fn = jax.vmap(self.fmt)   # 预先向量化
        self.num_params = self.param_size
        self.layout = ['u', 'u_x', 'u_xx', 'u_xxx', 'u_t']

        # 多项损失的权重（与模板一致）
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0

        # 保存 PDE/IC 参数
        self.V1, self.V2 = float(V1), float(V2)
        self.C1, self.C2 = float(C1), float(C2)
        self.X1, self.X2 = float(X1), float(X2)

        def ic_function(x):
            # x: (N,2) or (N,1) 取第一列为空间
            xv = x[:, :1]
            a1 = 0.5 * jnp.sqrt(self.C1 / self.V2)
            a2 = 0.5 * jnp.sqrt(self.C2 / self.V2)
            sech = lambda z: 1.0 / jnp.cosh(z)
            return 3.0 * self.C1 * sech(a1 * (xv - self.X1))**2 + 3.0 * self.C2 * sech(a2 * (xv - self.X2))**2

        bc_config = [{
            'component': 0,
            'function': ic_function,                    # 非零 IC 目标
            'bc': (lambda _, on_initial: on_initial),   # t=0 超面
            'type': 'ic'
        }]
        self.bcs = addbc(bc_config, self.geom_time)

        # --- 数据加载（照模板写法） ---


        def data_load(path):
            # 用 pandas 读取 CSV，自动跳过表头
            sim = pd.read_csv(path)
            # 拼成训练输入 [x, t]
            x_train = jnp.vstack([sim["x"].values, sim["t"].values]).T
            y_train = sim[["u"]].values
            # 可选：只截取空间区间 [0, 1.5]
            domain = (x_train[:, 0] >= self.bbox[0]) & (x_train[:, 0] <= self.bbox[1])
            X_ref, u_ref = x_train[domain], y_train[domain]

            return X_ref.astype(jnp.float32), u_ref.astype(jnp.float32)

        # --- 加载监督数据 ---
        X_ref, u_ref = data_load(datapath)
        self.X_data = X_ref
        self.Y_data = u_ref
        self.X_candidate = X_ref
        self.u_ref = u_ref

        # --- PDE采样器 (DataSampler_T) ---
        self.pde_data = DataSampler_T(self.geom_time, self.bcs, mul=4).train_x_all
        self.X_pde = self.pde_data
        self.Y_pde = np.zeros(shape=(len(self.X_pde), self.output_dim))

        # --- mini-batch ---
        self.BatchSize_eq = BatchSize_eq
        self.BatchSize_data = BatchSize_data
        domain_bounds = [
            [self.bbox[0], self.bbox[1]],  # x
            [self.bbox[2], self.bbox[3]],  # t
        ]
        self.pde_sampler = LowDiscrepancySampler(self.X_pde, self.Y_pde, domain_bounds)

        if len(self.X_data) > self.BatchSize_data:
            self.is_batch = True
            self.data_size = self.BatchSize_data
            self.data_sampler = LowDiscrepancySampler(self.X_data, self.Y_data, domain_bounds)
        else:
            self.is_batch = False
            self.data_size = len(self.X_data)

        # --- BC masks ---
        self.X_candidate = self.X_data
        self.u_ref = self.Y_data

        self.bcs_masks = [bc.filter(self.X_candidate) for bc in self.bcs]
        self.bcs_points = [self.X_candidate[mask] for mask in self.bcs_masks]

        # --- 定义 reset / step（固定写法） ---
        def reset_fn(key):
            X_eq, Y_eq = self.pde_sampler.get_batch(batch_size=self.BatchSize_eq)
            if self.is_batch:
                X_d, Y_d = self.data_sampler.get_batch(batch_size=self.BatchSize_data)
            else:
                X_d, Y_d = self.X_data, self.Y_data
            masks = [bc.filter(X_eq) for bc in self.bcs]
            X_batch = jnp.concatenate((X_eq, X_d), axis=0)
            Y_batch = jnp.concatenate((Y_eq, Y_d), axis=0)
            return State(obs=X_batch, labels=Y_batch, bcs_masks=masks)

        def step_fn(states, actions):
            def single_loss_fn(s, a):
                reward = -self.loss_fn(pred=a, X_batch=s.obs, Y_batch=s.labels, bcs_masks=s.bcs_masks)
                return reward

            rewards = jax.vmap(single_loss_fn)(states, actions)
            done = jnp.ones((actions.shape[0],), dtype=jnp.int32)
            return states, rewards, done

        self._reset_fn = reset_fn
        self._step_fn = step_fn
    # ---------- 参数初始化 / seed ----------
    def _init_params(self):
        key = random.PRNGKey(self.seed)
        dummy = jnp.zeros((1, self.input_dim))
        self.params_tree = self.net.init(key, dummy)
        self.param_size, self.fmt = get_params_format_fn(self.params_tree)

    def update_seed(self, seed):
        self.seed = seed
        self._init_params()

    # ---------- 关键：KdV 的 PDE 残差 ----------
    def pde_fn(self, pred):
        """根据网络输出 action 计算 KdV 方程残差：
           r = u_t + V1 * u * u_x + V2 * u_xxx
           说明：layout 与 self.layout 对齐 -> [u, u_x, u_xx, u_xxx, u_t]
        """
        u     = pred[:, 0:1]
        u_x   = pred[:, 1:2]
        u_xxx = pred[:, 3:4]
        u_t   = pred[:, 4:5]
        r_u = u_t + self.V1 * u * u_x + self.V2 * u_xxx
        return r_u

    def data_fn(self, Y_ref, pred, mask):
        u_true = Y_ref[-self.data_size:, 0:1]
        u_pred = pred[-self.data_size:, 0:1]
        loss_tmp = (u_pred - u_true) ** 2
        data_loss = jnp.sum(loss_tmp) / pred.shape[0]
        return data_loss

    def loss_fn(self, pred, X_batch, Y_batch, bcs_masks):
        pde_size = self.BatchSize_eq
        pde_pred = pred[:pde_size, :]
        X_pde = X_batch[:pde_size, :]

        # ---- PDE loss ----
        r_all = self.pde_fn(pde_pred)  # (N_pde, 1)
        comb_mask = jnp.any(jnp.stack(bcs_masks, axis=1), axis=1)
        interior_mask = (~comb_mask).astype(r_all.dtype)
        r_masked = r_all * interior_mask[:, None]
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior_mask) + 1e-8)

        # ---- IC / BC loss ----
        bc_loss_sum, ic_loss_sum = 0.0, 0.0
        bc_count, ic_count = 0, 0

        for bc, mask in zip(self.bcs, bcs_masks):
            mask_f = mask[:, None].astype(pde_pred.dtype)
            err = bc.error(pde_pred, X_pde)  # (N_pde, 1)
            term = jnp.sum((err ** 2) * mask_f) / (jnp.sum(mask_f) + 1e-8)

            if isinstance(bc, IC):
                ic_loss_sum += term
                ic_count += 1
            else:
                bc_loss_sum += term
                bc_count += 1

        ic_loss = ic_loss_sum / (ic_count + 1e-8)
        bc_loss = bc_loss_sum / (bc_count + 1e-8)

        # ---- Data loss ----
        data_loss = self.data_fn(Y_batch, pred, interior_mask)

        # ---- Total loss ----
        total_loss = (
            self.pde_lambda * pde_loss +
            self.ic_lambda * ic_loss +
            self.bc_lambda * bc_loss +
            self.data_lambda * data_loss
        )

        jax.debug.print("pde_loss={:}, ic_loss={:}, bc_loss={:}, data_loss={:}",
                        pde_loss, ic_loss, bc_loss, data_loss)

        return total_loss

    # ---------- VectorizedTask 接口 ----------
    def reset(self, key):
        return self._reset_fn(key)

    def step(self, state, action):
        return self._step_fn(state, action)

# ---------- Policy ----------
class PINNsPolicy(PolicyNetwork):
    def __init__(self, net, num_params, format_params_fn, grad_keys):
        self.net = net
        self.num_params = num_params
        self.format_params_fn = format_params_fn
        self.grad_keys = grad_keys

    def get_actions(self,
                    t_states: TaskState,
                    flat_params: jnp.ndarray,
                    p_states: PolicyState):
        # flat → pytree
        params_tree = self.format_params_fn(flat_params)  # batched pytree

        # 把 obs 第一维与 params_tree 对齐：obs_i 给 params_i
        obs = t_states.obs  # (B, N_pts, 2)

        # forward + derivatives
        def f_single(params, obs_i):
            outs = self.net.derivatives(params, obs_i)  # dict
            return stack_outputs(outs, self.grad_keys)  # (N_pts, 5)

        actions = jax.vmap(f_single)(params_tree, obs)  # (B, N_pts, 5)
        return actions, p_states

# ---------- Initialize task & policy ----------
seeds = 0
train_task = KdV1D()
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
    