import jax
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
from jax import random
import numpy as np
from src.data import DataSampler_T, LowDiscrepancySampler
from typing import Sequence

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

# ---------- 网络 ----------
class PINN(BaseNN):
    """
        PINN for 1D viscous Burgers:
            u_t + u * u_x - NU * u_xx = 0
        IC: u(x,0) = sin(-π x)
        BC: u(-1,t) = u(1,t) = 0
    """
    def derivatives(self, params, X):
        # 1) 标量 forward
        f_scalar = lambda z: jnp.ravel(self.apply(params, z[None, :]))[0]

        # 2) 一阶梯度 ∂u/∂(x,t)   (N,2)
        grads = jax.vmap(jax.grad(f_scalar))(X)  # (N,2)
        u_x, u_t = grads[:, :1], grads[:, 1:2]

        # 3) 二阶 ∂²u/∂x²         (N,1)
        u_xx = jax.vmap(lambda z: jax.hessian(f_scalar)(z)[0, 0])(X)[:, None]

        # 4) u itself            (N,1)
        u = jax.vmap(f_scalar)(X)[:, None]

        return {'u': u,
                'u_x': u_x,
                'u_t': u_t,
                'u_xx': u_xx}

class Burgers1D(VectorizedTask):
    def __init__(self, datapath =ref_dir / 'burgers1d.dat', geom=[-1, 1], time=[0, 1], nu=0.01 / np.pi):
        # --- 基础设置 ---
        self.max_steps = 1
        self.obs_shape = tuple([2, ])
        self.act_shape = tuple([1, ])

        # PDE参数
        self.nu = nu    # 粘性系数

        # 域定义
        self.geom = geometry.Interval(*geom)
        time_domain = geometry.TimeDomain(*time)
        self.geom_time = geometry.GeometryXTime(self.geom, time_domain)

        # --- 网络定义 ---
        self.output_dim = 1
        self.input_dim = self.geom_time.dim if self.geom_time is not None else self.geom.dim
        self.net = PINN(input_dim=self.input_dim, output_dim=self.output_dim)

        # 参数初始化（添加 seed 参数）
        self.seed = 0  # 默认值，可通过方法修改
        self._init_params()
        self.format_params_fn = jax.vmap(self.fmt)   # 预先向量化
        self.num_params = self.param_size
        self.layout = ['u', 'u_x', 'u_xx', 'u_t']

        # 多项损失值的权重
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0

        # --- 定义 IC & BC 配置 ---
        def f_ic(x):
            x=x[:,0:1]
            return jnp.sin(-jnp.pi * x)

        # IC & BC
        bc_config = [{
            'component': 0,
            'function': f_ic,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda x: jnp.zeros_like(x[:, 0:1])),
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet'
        }]
        self.bcs = addbc(bc_config, self.geom_time)

        # --- pde data load ---
        def data_load(path):
            loader = DataLoader()
            loader.load(path,
                        input_dim=2,  # x, t
                        output_dim=1,  # u
                        t_transpose=True)
            data = loader.ref_data  # numpy (N, 3)
            X_all = jnp.array(data[:, :2], jnp.float32)
            y_all = jnp.array(data[:, 2:], jnp.float32)
            # # 根据定义域掩码对数据进行过滤
            # mask = (
            #         (X_all[:, 0] >= geom[0]) & (X_all[:, 0] <= geom[1]) &
            #         (X_all[:, 1] >= time[0]) & (X_all[:, 1] <= time[1])
            # )
            return X_all, y_all

        # --- load data ---
        X_ref, u_ref = data_load(datapath)

        self.X_data = X_ref
        self.Y_data = u_ref

        # --- pde points ---
        self.pde_data = DataSampler_T(self.geom_time, self.bcs, mul=4).train_x_all
        self.X_pde = self.pde_data
        self.Y_pde = jnp.zeros_like(self.X_pde[:, :1])  # 目标是0

        # --- mini batch ---
        self.BatchSize_eq = BatchSize_eq
        self.BatchSize_data = BatchSize_data
        domain_bounds = [
            [geom[0], geom[1]],  # x_min, x_max
            [time[0], time[1]],  # t_min, t_max
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

        # --- 定义 reset / step ---
        def reset_fn(key):
            X_eq, Y_eq = self.pde_sampler.get_batch(batch_size=self.BatchSize_eq)  # 采样内部可能存在循环问题
            if self.is_batch:
                X_d, Y_d = self.data_sampler.get_batch(batch_size=self.BatchSize_data)
            else:
                X_d, Y_d = self.X_data, self.Y_data
            masks = [bc.filter(X_eq) for bc in self.bcs]
            X_batch = np.concatenate((X_eq, X_d), axis=0)
            Y_batch = np.concatenate((Y_eq, Y_d), axis=0)
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

    def _init_params(self):
        key = random.PRNGKey(self.seed)
        dummy = jnp.zeros((1, self.input_dim))
        self.params_tree = self.net.init(key, dummy)
        self.param_size, self.fmt = get_params_format_fn(self.params_tree)

    def update_seed(self, seed):
        self.seed = seed
        self._init_params()

    def pde_fn(self, pred):
        """根据网络输出action + 坐标计算 pde 残差"""
        u, u_x, u_xx, u_t = (
            pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
        )
        return u_t + u * u_x - self.nu * u_xx
    
    def data_fn(self, Y_ref, pred): 
        u_ref = Y_ref[:, 0:1]
        u_pred = pred[-Y_ref.shape[0]:, 0:1] 

        loss_tmp = (u_pred - u_ref) ** 2
        data_loss = jnp.sum(loss_tmp) / pred.shape[0]
        return data_loss

    def loss_fn(self, pred, X_batch, Y_batch, bcs_masks):
        pde_size = self.BatchSize_eq
        X_eq = X_batch[:pde_size]
        pde_pred = pred[:pde_size, :]
        data_pred = pred[pde_size:]
        data_true = Y_batch[pde_size:]

        comb_mask = jnp.any(jnp.stack([mask[:pde_size] for mask in bcs_masks], axis=1), axis=1)

        # PDE loss
        r_all = self.pde_fn(pde_pred)
        interior = (~comb_mask).astype(r_all.dtype)
        r_masked = r_all * interior[:, None]
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior) + 1e-8)

        # IC/BC loss
        ic_loss_sum = 0.0
        bc_loss_sum = 0.0
        ic_category = 0
        bc_category = 0
        for bc, mask in zip(self.bcs, bcs_masks):
            mask_eq = mask[:pde_size]
            mask_f = mask_eq[:, None].astype(pde_pred.dtype)
            err = bc.error(pde_pred, X_eq)
            term = jnp.sum((err ** 2) * mask_f) / (jnp.sum(mask_f) + 1e-8)
            if isinstance(bc, IC):
                ic_loss_sum += term
                ic_category += 1
            else:
                bc_loss_sum += term
                bc_category += 1
        ic_loss = ic_loss_sum / (ic_category + 1e-8)
        bc_loss = bc_loss_sum / (bc_category + 1e-8)

        data_loss = self.data_fn(data_true, data_pred)

        loss = (self.pde_lambda * pde_loss +
                self.ic_lambda * ic_loss +
                self.bc_lambda * bc_loss +
                self.data_lambda * data_loss)

        jax.debug.print("pde_loss={:}, ic_loss={:}, bc_loss={:}, data_loss={:}", pde_loss, ic_loss, bc_loss, data_loss)
        return loss

    def reset(self, key):
        return self._reset_fn(key)

    def step(self, state, action):
        return self._step_fn(state, action)


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
            return stack_outputs(outs, self.grad_keys)  # (N_pts, 4)

        actions = jax.vmap(f_single)(params_tree, obs)  # (B, N_pts, 4)
        return actions, p_states


# # Initialize task & policy
seeds = 0
train_task = Burgers1D()
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
