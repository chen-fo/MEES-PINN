import os
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
from flax import linen as nn
from jax import random, vmap, hessian, jacfwd
import numpy as np

from EAPINN import geometry
from EAPINN.ICBC import IC
from ..utils import DataLoader, SimManager, addbc


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray


node = 8  # 可以修改为更深或更宽的网络
datapath = os.path.join(os.path.dirname(__file__), '..', '..', 'ref', 'burgers1d.dat')


class PINNs(nn.Module):
    def setup(self, ):
        # 五层全连接 + tanh
        self.layers = [
            nn.Dense(node, kernel_init=jax.nn.initializers.glorot_uniform()),
            nn.tanh,
            nn.Dense(node, kernel_init=jax.nn.initializers.glorot_uniform()),
            nn.tanh,
            nn.Dense(node, kernel_init=jax.nn.initializers.glorot_uniform()),
            nn.tanh,
            nn.Dense(node, kernel_init=jax.nn.initializers.glorot_uniform()),
            nn.tanh,
            nn.Dense(1, kernel_init=jax.nn.initializers.glorot_uniform())
        ]

    @nn.compact
    def __call__(self, inputs):
        # inputs: [batch, 2] -> (x, t)
        x, t = inputs[:, 0:1], inputs[:, 1:2]

        # 网络前向得到 u(x,t)
        def get_u(xi, ti):
            u = jnp.hstack([xi, ti])
            for lyr in self.layers:
                u = lyr(u)
            return u

        def get_u_t(get_u, xi, ti):
            u_t = jacfwd(get_u, 1)(xi, ti)
            return u_t

        def get_u_x(get_u, xi, ti):
            u_x = jacfwd(get_u)(xi, ti)
            return u_x

        def get_u_xx(get_u, xi, ti):
            u_xx = hessian(get_u)(xi, ti)
            return u_xx

        # 依次计算各导数
        u = get_u(x, t)

        u_t_vamp = vmap(get_u_t, in_axes=(None, 0, 0))
        u_t = u_t_vamp(get_u, x, t).reshape(-1, 1)

        u_x_vamp = vmap(get_u_x, in_axes=(None, 0, 0))
        u_x = u_x_vamp(get_u, x, t).reshape(-1, 1)

        u_xx_vamp = vmap(get_u_xx, in_axes=(None, 0, 0))
        u_xx = u_xx_vamp(get_u, x, t).reshape(-1, 1)

        action = jnp.hstack([u, u_x, u_xx, u_t])

        return action


class Burgers1D(VectorizedTask):
    """
    PINN for 1D viscous Burgers:
        u_t + u * u_x - NU * u_xx = 0
    IC: u(x,0) = sin(-π x)
    BC: u(-1,t) = u(1,t) = 0
    """

    def __init__(self, datapath=datapath, geom=[-1, 1], time=[0, 1], nu=0.01 / np.pi):
        # --- 基础设置 ---
        self.max_steps = 1
        self.obs_shape = tuple([2, ])
        self.act_shape = tuple([1, ])

        # PDE参数
        self.nu = nu  # 粘性系数

        # 域定义
        self.geom = geometry.Interval(*geom)
        time_domain = geometry.TimeDomain(*time)
        self.geom_time = geometry.GeometryXTime(self.geom, time_domain)

        # 多项损失值的权重
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0

        # --- 定义 IC & BC 配置 ---
        def f_ic(x):
            return jnp.sin(-jnp.pi * x)

        # IC & BC
        bc_config = [{
            'component': 0,
            'function': f_ic,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _: 0),
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

            # 根据定义域掩码对数据进行过滤
            mask = (
                    (X_all[:, 0] >= geom[0]) & (X_all[:, 0] <= geom[1]) &
                    (X_all[:, 1] >= time[0]) & (X_all[:, 1] <= time[1])
            )
            return X_all[mask], y_all[mask]

        X_ref, u_ref = data_load(datapath)
        self.X_candidate = X_ref
        self.u_ref = u_ref

        self.bcs_masks = [bc.filter(self.X_candidate) for bc in self.bcs]
        self.bcs_points = [X_ref[mask] for mask in self.bcs_masks]

        # --- 定义 reset / step ---
        def reset_fn(key):
            batch_data, batch_labels = X_ref, u_ref
            return State(obs=batch_data, labels=batch_labels)

        def step_fn(state, action):
            reward = - self.loss_fn(action)
            done = jnp.ones((), dtype=jnp.int32)
            return state, reward, done

        # 向量化 + JIT
        self._reset_fn = jax.jit(jax.vmap(reset_fn))
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def pde_fn(self, pred):
        """根据网络输出 + 坐标计算 pde 残差"""
        u, u_x, u_xx, u_t = (
            pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
        )
        return u_t + u * u_x - self.nu * u_xx

    def loss_fn(self, pred):
        X_all = self.X_candidate

        comb_mask = jnp.any(jnp.stack(self.bcs_masks, axis=1), axis=1)

        # 2) PDE loss（仅在内部点）
        r_all = self.pde_fn(pred)
        interior = (~comb_mask).astype(r_all.dtype)  # shape (N,)
        # 广播 interior 到 (N,1)，其它点 residual 会被置 0
        r_masked = r_all * interior[:, None]  # shape (N,1)
        # 求和再除以内部点数量，实现内点上的 MSE
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior) + 1e-8)

        # 3) IC & BC loss：分别按类型累积
        ic_loss = 0.0
        bc_loss = 0.0
        for bc, mask, X_bc in zip(self.bcs, self.bcs_masks, self.bcs_points):
            M = X_bc.shape[0]
            if M == 0:
                continue
            # 从 pred 中先切出 (M,4) 的子集
            pred_bc = pred[mask]
            # 调用新的 error 接口
            err = bc.error(pred_bc, X_bc)  # (M,1)
            term = jnp.mean(err ** 2)
            if isinstance(bc, IC):
                ic_loss += term
            else:
                bc_loss += term

        loss = (self.pde_lambda * pde_loss +
                self.ic_lambda * ic_loss +
                self.bc_lambda * bc_loss)

        return loss

    def reset(self, key):
        return self._reset_fn(key)

    def step(self, state, action):
        return self._step_fn(state, action)


class PINNsPolicy(PolicyNetwork):
    def __init__(self, seed=0):
        model = PINNs()
        key1, key2 = random.split(random.PRNGKey(seed))
        dummy = jnp.zeros((1, 2))
        params = model.init(key1, dummy)
        # 扁平化 & 重塑函数
        self.num_params, fmt = get_params_format_fn(params)
        self._format_params_fn = jax.vmap(fmt)
        # jit + vmap 前向
        self._forward_fn = jax.jit(jax.vmap(model.apply))

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState):
        p = self._format_params_fn(params)
        out = self._forward_fn(p, t_states.obs)
        return out, p_states


# Initialize task & policy
seed = 0
train_task = Burgers1D()
policy = PINNsPolicy()
sim_mgr = SimManager(n_repeats=1, test_n_repeats=1, pop_size=0, n_evaluations=1,
                     policy_net=policy, train_vec_task=train_task, valid_vec_task=train_task,
                     seed=seed)


def get_fitness(samples):
    scores, _ = sim_mgr.eval_params(params=samples, test=False)
    return scores
