import jax
from jax import random
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
import numpy as np
from scipy.interpolate import griddata
from typing import Sequence

from EAPINN import geometry
from EAPINN.ICBC import IC
from src.utils import SimManager, addbc, stack_outputs, DataLoader
from src.data import DataSampler, LowDiscrepancySampler
from src.nn import BaseNN
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent  # 三级父目录就是项目根目录
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
    PINN for Wave2D_Heterogeneous :
    PDE:
        ∂²u/∂x² + ∂²u/∂y² − (1/coef(x,y))·∂²u/∂t² = 0，
        其中 coef(x,y) 是基于 Darcy 系数场插值得到的空间异质系数

    IC:
        u(x, y, 0) = exp(−((x−μ₁)² + (y−μ₂)²)/(2 σ²)),   μ = (−0.5, 0), σ = 0.3
        ∂u/∂t(x, y, 0) = 0

    BC (Neumann, spatial):
        ∂u/∂n(x, y, t) = 0,   (x, y) ∈ ∂Ω (矩形边界)
    """
    def derivatives(self, params, X):
        def forward(z):
            return self.apply(params, z[None, :])[0]

        def u_fn(z): return jnp.squeeze(forward(z))

        u = jax.vmap(u_fn)(X).reshape(-1, 1)
        grads_u = jax.vmap(jax.grad(u_fn))(X)
        hess_u = jax.vmap(jax.hessian(u_fn))(X)
        
        return {
            'u': u,
            'u_x': grads_u[:, 0:1], 'u_y': grads_u[:, 1:2], 'u_t': grads_u[:, 1:2],
            'u_xx': hess_u[:, 0, 0:1], 'u_yy': hess_u[:, 1, 1:2], 'u_tt': hess_u[:, 2, 2:3],
        }


class Wave2D_Heterogeneous(VectorizedTask):
    def __init__(self, datapath=ref_dir/'wave_darcy.dat', bbox=[-1, 1, -1, 1, 0, 5], mu=(-0.5, 0), sigma=0.3):
        # --- 基础设置 ---
        self.max_steps = 1
        self.obs_shape = tuple([3, ])
        self.act_shape = tuple([1, ])

        # PDE参数
        self.mu = mu
        self.sigma = sigma
        self.darcy_2d_coef = np.loadtxt("../ref/darcy_2d_coef_256.dat")
        #self.coef_array = None

        # 域定义
        self.bbox = bbox
        self.geom = geometry.Hypercube(xmin=(self.bbox[0], self.bbox[2], self.bbox[4]), xmax=(self.bbox[1], self.bbox[3], self.bbox[5]))
        self.geom_time = None

        # --- 网络定义 ---
        self.output_dim = 1
        self.input_dim = self.geom_time.dim if self.geom_time is not None else self.geom.dim
        self.net = PINN(input_dim=self.input_dim, output_dim=self.output_dim)

        # 参数初始化（添加 seed 参数）
        self.seed = 0  # 默认值，可通过方法修改
        self._init_params()
        self.format_params_fn = jax.vmap(self.fmt)   # 预先向量化
        self.num_params = self.param_size
        self.layout = ['u', 'u_x', 'u_y', 'u_t', 'u_xx', 'u_yy', 'u_tt']

        # 多项损失值的权重
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0

        # --- 定义 IC & BC 配置 ---
        def ic_func(x):
            return jnp.exp(-((x[:, 0:1] - self.mu[0])**2 + (x[:, 1:2] - self.mu[1])**2) / (2 * self.sigma**2))

        def boundary_t0(x, on_initial):
            return jnp.isclose(x[:, 2], self.bbox[4])

        def boundary_rec(x, on_boundary):
            x0, x1 = x[:, 0], x[:, 1]
            is_corner = (
                (jnp.isclose(x0, bbox[0]) | jnp.isclose(x0, bbox[1])) &
                (jnp.isclose(x1, bbox[2]) | jnp.isclose(x1, bbox[3]))
            )
            return jnp.logical_and(on_boundary, ~is_corner)

        bc_config = [
            {
                'component': 0,
                'function': ic_func,
                'bc': boundary_t0,
                'type': 'dirichlet'
            },
            {
                'component': 0,
                'function': (lambda _: 0),
                'bc': boundary_t0,
                'type': 'neumann'
            },
            {
                'component': 0,
                'function': (lambda _: 0),
                'bc': boundary_rec,
                'type': 'neumann',
            }
        ]

        self.bcs = addbc(bc_config, self.geom)

        # --- pde points ---
        self.pde_data = DataSampler(self.geom, self.bcs, mul=4).train_x_all
        X_pde = self.pde_data

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
        coef_all = np.array(griddata(self.darcy_2d_coef[:, 0:2], self.darcy_2d_coef[:, 2], (X_pde[:, 0:2] + 1) / 2)).reshape(-1, 1)
        self.Y_pde = coef_all
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

    def pde_fn(self, pred, coef, X=None):
        u = pred[:, 0:1]
        u_xx, u_yy, u_tt = pred[:, 4:5], pred[:, 5:6], pred[:, 6:7]

        r = u_xx + u_yy - u_tt / (coef + 1e-8)
        return r

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

        r_all = self.pde_fn(pde_pred, Y_pde)
        interior = (~comb_mask).astype(r_all.dtype)
        r_masked = r_all * interior[:, None]
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior))

        # IC & BC loss
        ic_loss_sum = 0.0
        bc_loss_sum = 0.0
        ic_category = 0
        bc_category = 0
        for bc, mask in zip(self.bcs, bcs_masks):

            mask_f = mask[:, None].astype(pde_pred.dtype)
            err = bc.error(pde_pred, X_pde)
            term = jnp.sum((err ** 2) * mask_f) / (jnp.sum(mask_f) + 1e-8)
            if isinstance(bc, IC):
                ic_loss_sum += term
                ic_category += 1
            else:
                bc_loss_sum += term
                bc_category += 1
        ic_loss = ic_loss_sum / (ic_category + 1e-8)
        bc_loss = bc_loss_sum / (bc_category + 1e-8)

        # data loss
        data_loss = self.data_fn(Y_batch, pred)

        loss = (self.pde_lambda * pde_loss +
                self.ic_lambda * ic_loss +
                self.bc_lambda * bc_loss +
                self.data_lambda * data_loss)

        jax.debug.print("pde_loss={:},ic_loss={:},bc_loss={:},data_loss={:}", pde_loss, ic_loss, bc_loss, data_loss)
        return loss


    def coef(self, X):
        X_ = jnp.array((X[:, 0:2] + 1) / 2)
        return griddata(self.darcy_2d_coef[:, 0:2], self.darcy_2d_coef[:, 2], X_)

    def reset(self, key):
        # # 后续考虑将coef作为一列加在X_batch后面，然后采样的时候就可以用
        # new_state = self._reset_fn(key)
        # X_eq = new_state.obs[:self.BatchSize_eq, :]
        # self.coef_array = jnp.array(griddata(self.darcy_2d_coef[:, 0:2], self.darcy_2d_coef[:, 2], (X_eq[:, 0:2] + 1) / 2)).reshape(-1, 1)
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
        params_tree = self.format_params_fn(flat_params)  # batched pytree

        obs = t_states.obs

        # forward + derivatives
        def f_single(params, obs_i):
            outs = self.net.derivatives(params, obs_i)
            return stack_outputs(outs, self.grad_keys)

        actions = jax.vmap(f_single)(params_tree, obs)
        return actions, p_states

# # Initialize task & policy
seeds = 0
train_task = Wave2D_Heterogeneous()
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
