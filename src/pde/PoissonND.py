import jax
from jax import random
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
import numpy as np
from typing import Sequence

from EAPINN import geometry
from EAPINN.ICBC import IC
from src.utils import SimManager, addbc, stack_outputs
from src.data import DataSampler, LowDiscrepancySampler
from src.nn import BaseNN

BatchSize = 4096


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray
    bcs_masks: Sequence[jnp.ndarray]


# ---------- 网络 ----------
class PINN(BaseNN):
    """
    PINN for PoissonND :
    PDE:
        Σ_{i=1}^d ∂²u/∂x_i² + (π²/4) Σ_{i=1}^d sin(π/2·x_i) = 0

    BC (Dirichlet BC):
        u(x) = Σ_{i=1}^d sin(π/2·x_i),   x ∈ ∂Ω
    """
    def derivatives(self, params, X):
        d = X.shape[1]

        def forward(z):
            return self.apply(params, z[None, :])[0]

        u_fn = lambda z: jnp.squeeze(forward(z))

        u = jax.vmap(u_fn)(X)[:, None]
        grad_u = jax.vmap(jax.grad(u_fn))(X)
        hess_u = jax.vmap(jax.hessian(u_fn))(X)


        diag_h = jnp.diagonal(hess_u, axis1=1, axis2=2)  # (N,d+1)
        laplace = jnp.sum(diag_h[:, :d], axis=1, keepdims=True)  # (N,1)

        return {
            'u': u,
            'laplace_u': laplace,
        }


class PoissonND(VectorizedTask):
    def __init__(self, dim=5, length=1):
        # --- 基础设置 ---
        self.max_steps = 1
        self.obs_shape = tuple([3, ])
        self.act_shape = tuple([1, ])

        # PDE参数
        self.dim = dim

        # 域定义
        self.bbox = [0, length] * self.dim
        self.geom = geometry.Hypercube(xmin=self.bbox[0::2], xmax=self.bbox[1::2])
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
        self.layout = ['u', 'laplace_u']

        # 多项损失值的权重
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 0.0

        # --- 定义 IC & BC 配置 ---
        def ref_sol(x):
            return jnp.sin(np.pi / 2 * x).sum(axis=1).reshape(-1, 1)

        bc_config =[
            {
                'component': 0,
                'function': ref_sol,
                'bc': (lambda _, on_boundary: on_boundary),
                'type': 'dirichlet',
            }
        ]

        self.bcs = addbc(bc_config, self.geom)

        # --- pde points ---
        self.pde_data = DataSampler(self.geom, self.bcs, mul=4).train_x_all
        X_ref = self.pde_data
        Y_ref = ref_sol(X_ref)

        self.X_all = X_ref
        self.Y_all = Y_ref
        self.X_candidate = X_ref
        self.u_ref = Y_ref
        # --- mini batch ---
        self.batch_size = BatchSize
        domain_bounds = []
        for i in range(0, len(self.bbox), 2):
            domain_bounds.append([self.bbox[i], self.bbox[i + 1]])

        self.sampler = LowDiscrepancySampler(self.X_all, self.Y_all, domain_bounds)

        # --- 定义 reset / step ---
        def reset_fn(key):
            X_batch, Y_batch = self.sampler.get_batch(batch_size=BatchSize)  # 采样内部可能存在循环问题
            masks = [bc.filter(X_batch) for bc in self.bcs]
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
        u_xx = pred[:, 1:2]

        def f_src(x):
            return (jnp.pi ** 2) / 4 * jnp.sin(jnp.pi / 2 * x).sum(axis=1).reshape(-1, 1)

        r = u_xx + f_src(X)

        return r

    def data_fn(self, Y_ref, pred, mask):
        u_ref = Y_ref[:, 0:1]
        u_pred = pred[:, 0:1]
        loss_tmp = (u_pred - u_ref) ** 2

        # data_loss = jnp.sum(loss_tmp ** 2) / pred.shape[0]
        loss_masked = loss_tmp * mask[:, None]
        data_loss = jnp.sum(loss_masked) / (jnp.sum(mask))
        return data_loss

    def loss_fn(self, pred, X_batch, Y_batch, bcs_masks):
        comb_mask = jnp.any(jnp.stack(bcs_masks, axis=1), axis=1)

        # PDE loss
        r_all = self.pde_fn(pred, X_batch)
        interior = (~comb_mask).astype(r_all.dtype)
        r_masked = r_all * interior[:, None]
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior))

        # IC & BC loss
        ic_loss_sum = 0.0
        bc_loss_sum = 0.0
        ic_category = 0
        bc_category = 0
        for bc, mask in zip(self.bcs, bcs_masks):

            mask_f = mask[:, None].astype(pred.dtype)
            err = bc.error(pred, X_batch)
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
        data_loss = 0

        loss = (self.pde_lambda * pde_loss +
                self.ic_lambda * ic_loss +
                self.bc_lambda * bc_loss +
                self.data_lambda * data_loss)

        jax.debug.print("pde_loss={:},ic_loss={:},bc_loss={:},data_loss={:}", pde_loss, ic_loss, bc_loss, data_loss)
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
train_task = PoissonND()
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

