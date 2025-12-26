import jax
from jax import random
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
import numpy as np
import scipy
from typing import Sequence

from EAPINN import geometry
from EAPINN.ICBC import IC
from src.utils import addbc, stack_outputs, DataLoader
from src.data import DataSampler_T, LowDiscrepancySampler
from src.nn import BaseNN
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
ref_dir = project_root / 'ref'

BatchSize_eq = 2048
BatchSize_data = 2048

@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray
    bcs_masks: Sequence[jnp.ndarray]

class PINN(BaseNN):
    """
    PINN for Burgers2D :
    PDE:
        ∂u₁/∂t + u₁·∂u₁/∂x + u₂·∂u₁/∂y − ν·(∂²u₁/∂x² + ∂²u₁/∂y²) = 0
        ∂u₂/∂t + u₁·∂u₂/∂x + u₂·∂u₂/∂y − ν·(∂²u₂/∂x² + ∂²u₂/∂y²) = 0

    IC:
        u₁(x, y, 0) = ic₁(x, y)
        u₂(x, y, 0) = ic₂(x, y)

    BC (Periodic in x and y):
        u(0,   y, t) = u(L,   y, t)
        u(L,   y, t) = u(0,   y, t)
        u(x,   0, t) = u(x,   L, t)
        u(x,   L, t) = u(x,   0, t)
    """
    def derivatives(self, params, X):
        def forward(z):
            return self.apply(params, z[None, :])[0]

        def u_fn(z): return forward(z)[0]

        def v_fn(z): return forward(z)[1]

        u = jax.vmap(u_fn)(X).reshape(-1, 1)
        v = jax.vmap(v_fn)(X).reshape(-1, 1)
        grads_u = jax.vmap(jax.grad(u_fn))(X)   # ∂u/∂x, ∂u/∂y, ∂u/∂t
        grads_v = jax.vmap(jax.grad(v_fn))(X)   # ∂v/∂x, ∂v/∂y, ∂v/∂t
        hess_u = jax.vmap(jax.hessian(u_fn))(X)
        hess_v = jax.vmap(jax.hessian(v_fn))(X)

        return {
            'u': u,
            'v': v,
            'u_x': grads_u[:, 0:1], 'u_y': grads_u[:, 1:2], 'u_t': grads_u[:, 2:3],
            'u_xx': hess_u[:, 0, 0:1], 'u_yy': hess_u[:, 1, 1:2],
            'v_x': grads_v[:, 0:1], 'v_y': grads_v[:, 1:2], 'v_t': grads_v[:, 2:3],
            'v_xx': hess_v[:, 0, 0:1], 'v_yy': hess_v[:, 1, 1:2]
        }


class PDE(VectorizedTask):
    def __init__(self, hidden_layers=None, datapath=ref_dir / 'burgers2d_0.dat',
                 icpath=(ref_dir / 'burgers2d_init_u_0.dat', ref_dir / 'burgers2d_init_v_0.dat'),
                 L=4, T=1, nu=0.001):
        self.max_steps = 1
        self.obs_shape = tuple([3, ])
        self.act_shape = tuple([2, ])

        # PDE参数
        self.nu = nu    # 粘性系数

        # 域定义
        self.bbox = [0, L, 0, L, 0, T]
        self.geom = geometry.Rectangle(self.bbox[0:4:2], self.bbox[1:4:2])
        time_domain = geometry.TimeDomain(self.bbox[4], self.bbox[5])
        self.geom_time = geometry.GeometryXTime(self.geom, time_domain)

        # --- 网络定义 ---
        self.output_dim = 2
        self.input_dim = self.geom_time.dim if self.geom_time is not None else self.geom.dim
        if hidden_layers is not None:
            parts = hidden_layers.split('*')
            width, depth = parts
            self.net = PINN(width=int(width), depth=int(depth), input_dim=self.input_dim, output_dim=self.output_dim)
        else:
            self.net = PINN(input_dim=self.input_dim, output_dim=self.output_dim)

        self.seed = 0
        self._init_params()
        self.format_params_fn = jax.vmap(self.fmt)
        self.num_params = self.param_size
        self.layout = ['u', 'v', 'u_x', 'u_y', 'u_t', 'u_xx', 'u_yy', 'v_x', 'v_y', 'v_t', 'v_xx', 'v_yy']

        # 多项损失值的权重
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0

        # --- 定义 IC & BC 配置 ---
        self.ics = (np.loadtxt(icpath[0]), np.loadtxt(icpath[1]))

        def ic_func(x, component):
            return scipy.interpolate.LinearNDInterpolator(self.ics[component][:, :2], self.ics[component][:, 2:])(x[:, :2])

        def boundary_ic(x, on_initial):
            time_cond = jnp.isclose(x[2], 0.0)
            return jnp.logical_and(on_initial, time_cond)

        def boundary_xb(x, on_boundary):
            cond1 = jnp.isclose(x[:, 0], 0.0)
            cond2 = jnp.isclose(x[:, 0], float(L))
            x_boundary = jnp.logical_or(cond1, cond2)
            return jnp.logical_and(on_boundary, x_boundary)

        def boundary_yb(x, on_boundary):
            cond1 = jnp.isclose(x[:, 1], 0.0)
            cond2 = jnp.isclose(x[:, 1], float(L))
            y_boundary = jnp.logical_or(cond1, cond2)
            return jnp.logical_and(on_boundary, y_boundary)

        def ic_masks_load(X):
            ics_masks = []
            for bc in self.bcs:
                if isinstance(bc, IC):
                    ics_masks.append(bc.filter(X))
                else:
                    continue
            return ics_masks

        bc_config = [
            {
                'component': 0,
                'function': (lambda x: ic_func(x, component=0)),
                'bc': boundary_ic,
                'type': 'ic'
            },
            {
                'component': 1,
                'function': (lambda x: ic_func(x, component=1)),
                'bc': boundary_ic,
                'type': 'ic'
            },
            {
                'component': 0,
                'type': 'periodic',
                'component_x': 0,
                'bc': boundary_xb,
            },
            {
                'component': 1,
                'type': 'periodic',
                'component_x': 0,
                'bc': boundary_xb,
            },
            {
                'component': 0,
                'type': 'periodic',
                'component_x': 1,
                'bc': boundary_yb,
            },
            {
                'component': 1,
                'type': 'periodic',
                'component_x': 1,
                'bc': boundary_yb,
            },
        ]

        self.bcs = addbc(bc_config, self.geom_time)
        self.bcs_masks = None
        self.bcs_points = None

        # --- pde points ---
        self.pde_data = DataSampler_T(self.geom_time, self.bcs, mul=4).train_x_all
        X_pde = self.pde_data
        self.ic_all_masks = ic_masks_load(X_pde)
        self.ic_all_points = [X_pde[mask] for mask in self.ic_all_masks]
        self.ic_u = ic_func(X_pde[self.ic_all_masks[0]], 0)
        self.ic_v = ic_func(X_pde[self.ic_all_masks[0]], 1)
        self.Y_ic = np.hstack([self.ic_u, self.ic_v])

        # --- data points ---
        def data_load(path):
            loader = DataLoader()
            loader.load(path, input_dim=self.input_dim, output_dim=self.output_dim, t_transpose=True)
            Data = loader.ref_data
            X_data = jnp.array(Data[:, :self.input_dim], jnp.float32)
            Y_data = jnp.array(Data[:, self.input_dim:], jnp.float32)

            return X_data, Y_data

        self.X_data, self.Y_data = data_load(datapath)

        self.X_pde = X_pde
        self.Y_pde = np.zeros(shape=(len(X_pde), self.output_dim))
        self.Y_pde[self.ic_all_masks[0]] = self.Y_ic
        self.X_candidate = self.X_data
        self.u_ref = self.Y_data

        # --- mini batch ---
        self.BatchSize_eq = BatchSize_eq
        self.BatchSize_data = BatchSize_data
        domain_bounds = [
            [self.bbox[0], self.bbox[1]],  # [x_min,x_max]
            [self.bbox[2], self.bbox[3]],  # [y_min,y_max]
            [self.bbox[4], self.bbox[5]],  # [t_min,t_max]
        ]
        self.pde_sampler = LowDiscrepancySampler(self.X_pde, self.Y_pde, domain_bounds)
        if len(self.X_data) > self.BatchSize_data:
            self.is_batch = True
            self.data_size = self.BatchSize_data
            self.data_sampler = LowDiscrepancySampler(self.X_data, self.Y_data, domain_bounds)
        else:
            self.is_batch = False
            self.data_size = len(self.X_data)

        # --- 定义 reset / step ---
        def reset_fn(key):
            X_eq, Y_eq = self.pde_sampler.get_batch(batch_size=self.BatchSize_eq)  # 采样内部可能存在循环问题
            masks_tmp = [bc.filter(X_eq) for bc in self.bcs]

            # add extra right bc point for period boundary condition
            X_extra = self.add_right_bc(X_eq, self.bcs, masks_tmp)
            Y_extra = np.zeros(shape=(len(X_extra), self.output_dim))

            X_eq_all = np.concatenate((X_eq, X_extra), axis=0)
            Y_eq_all = np.concatenate((Y_eq, Y_extra), axis=0)

            masks = [bc.filter(X_eq_all) for bc in self.bcs]

            if self.is_batch:
                X_d, Y_d = self.data_sampler.get_batch(batch_size=self.BatchSize_data)
            else:
                X_d, Y_d = self.X_data, self.Y_data

            X_batch = np.concatenate((X_eq_all, X_d), axis=0)
            Y_batch = np.concatenate((Y_eq_all, Y_d), axis=0)

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

    def pde_fn(self, pred, X=None):
        """根据网络输出action + 坐标计算 pde 残差"""
        u, v = pred[:, 0:1], pred[:, 1:2]
        u_x, u_y, u_t = pred[:, 2:3], pred[:, 3:4], pred[:, 4:5]
        u_xx, u_yy = pred[:, 5:6], pred[:, 6:7]
        v_x, v_y, v_t = pred[:, 7:8], pred[:, 8:9], pred[:, 9:10]
        v_xx, v_yy = pred[:, 10:11], pred[:, 11:12]

        r_u = u_t + u * u_x + v * u_y - self.nu * (u_xx + u_yy)
        r_v = v_t + u * v_x + v * v_y - self.nu * (v_xx + v_yy)

        return jnp.hstack([r_u, r_v])

    def data_fn(self, Y_ref, pred, mask=None):
        u_true = Y_ref[- self.data_size:, 0:1]
        u_pred = pred[- self.data_size:, 0:1]
        loss_tmp = (u_pred - u_true) ** 2

        data_loss = jnp.sum(loss_tmp) / pred.shape[0]
        return data_loss

    def loss_fn(self, pred, X_batch, Y_batch, bcs_masks):
        pde_size = self.BatchSize_eq
        pde_pred = pred[:pde_size, :]
        X_pde = X_batch[:pde_size, :]
        Y_pde = Y_batch[:pde_size, :]
        comb_mask = jnp.any(jnp.stack(bcs_masks, axis=1), axis=1)

        # PDE loss
        r_all = self.pde_fn(pde_pred)
        interior = (~comb_mask).astype(r_all.dtype)
        r_masked = r_all * interior[:, None]
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior))

        # IC & BC loss
        ic_loss_sum = 0.0
        bc_loss_sum = 0.0
        ic_category = 0
        bc_category = 0

        for bc, mask, X_bc in zip(self.bcs, self.bcs_masks, self.bcs_points):
            if isinstance(bc, IC):
                err = pde_pred[mask, ic_category:ic_category + 1] - Y_pde[mask, ic_category].reshape(-1, 1)
                term = jnp.mean(err ** 2)
                ic_loss_sum += term
                ic_category += 1
            else:
                err = bc.error(pde_pred[mask], X_bc)
                term = (jnp.mean(err ** 2))
                bc_loss_sum += term
                bc_category += 1
        ic_loss = ic_loss_sum / (ic_category + 1e-8)
        bc_loss = bc_loss_sum / (bc_category + 1e-8)

        # data loss
        data_loss = self.data_fn(Y_batch, pred)

        loss = jnp.hstack([self.pde_lambda * pde_loss, self.ic_lambda * ic_loss, self.bc_lambda * bc_loss, self.data_lambda * data_loss])

        return loss

    @staticmethod
    def add_right_bc(X_eq, bcs, bcs_masks):
        extra_points_list = []
        for bc, mask in zip(bcs, bcs_masks):
            if not isinstance(bc, IC):
                X_left = X_eq[mask]
                X_right = bc.geom.periodic_point(X_left, bc.component_x)
                extra_points_list.append(X_right)   #only add right points
        extra_points_list.pop(1), extra_points_list.pop(2)
        extra_points = np.vstack(extra_points_list)
        return extra_points

    def reset(self, key):
        new_state = self._reset_fn(key)
        self.BatchSize_eq = new_state.obs.shape[0] - self.BatchSize_data
        X_eq = new_state.obs[:self.BatchSize_eq, :]
        bcs_masks = new_state.bcs_masks
        self.bcs_masks = bcs_masks
        self.bcs_points = [X_eq[mask] for mask in bcs_masks]
        return new_state

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
        params_tree = self.format_params_fn(flat_params)  # batched pytree

        obs = t_states.obs

        # forward + derivatives
        def f_single(params, obs_i):
            outs = self.net.derivatives(params, obs_i)
            return stack_outputs(outs, self.grad_keys)

        actions = jax.vmap(f_single)(params_tree, obs)
        return actions, p_states
