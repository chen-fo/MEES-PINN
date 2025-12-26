import jax
from jax import random
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.util import get_params_format_fn
from src.data import DataSampler_T,LowDiscrepancySampler 
from typing import Sequence

import numpy as np

from EAPINN import geometry
from EAPINN.ICBC import IC
from src.utils import DataLoader, SimManager, addbc, stack_outputs, CSGMultiDifference
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
        PINN for Heat2D_ComplexGeometry:
        PDE:
            ∂u/∂t - ∂²u/∂x² - ∂²u/∂y² = 0
        
        IC:
            u(x, y, 0) = 0
        
        BC:
            ∂u/∂n = 5 - u
            ∂u/∂n = 1 - u
            ∂u/∂n = 0.1 - u
    """

    def derivatives(self, params, X):
        def forward(z):
            out = self.apply(params, z[None, :])
            return out[0]

        def u_fn(z): return forward(z)[0]

        # 计算 u 的一阶和二阶导数
        grads_u = jax.vmap(jax.grad(u_fn))(X)
        hess_u = jax.vmap(jax.hessian(u_fn))(X)

        # 获取 u 的预测值
        u = jax.vmap(u_fn)(X).reshape(-1, 1)

        # 返回 u 及其一阶、二阶导数
        return {
                'u': u,
                'u_xx': hess_u[:, 0, 0].reshape(-1, 1),  # ∂²u/∂x²
                'u_yy': hess_u[:, 1, 1].reshape(-1, 1),  # ∂²u/∂y²
                'u_t': grads_u[:, 2:3],  # ∂u/∂t
            }


class Heat2D_ComplexGeometry(VectorizedTask):
    def __init__(self, datapath =ref_dir/'heat_complex.dat', bbox=[-8, 8, -12, 12, 0, 3]):
        # --- 基础设置 ---
        self.max_steps = 1
        self.obs_shape = tuple([3, ])
        self.act_shape = tuple([1, ])

        # 域定义
        self.bbox = bbox
        rec = geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        # big circles
        big_circles = []
        big_centers = [(-4, -3), (4, -3), (-4, 3), (4, 3), (-4, -9), (4, -9), (-4, 9), (4, 9), (0, 0), (0, 6), (0, -6)]
        for center in big_centers:
            big_circles.append(geometry.Disk(center, radius=1))
        # small circles
        small_circles = []
        small_centers = [(-3.2, -6), (-3.2, 6), (3.2, -6), (3.2, 6), (-3.2, 0), (3.2, 0)]
        for center in small_centers:
            small_circles.append(geometry.Disk(center, radius=0.4))

        self.geom = CSGMultiDifference(rec, big_circles + small_circles)
        time_domain = geometry.TimeDomain(bbox[4], bbox[5])
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
        self.layout = ['u', 'u_xx', 'u_yy', 'u_t']

        # 多项损失值的权重
        self.pde_lambda = 1.0
        self.bc_lambda = 1.0
        self.ic_lambda = 1.0
        self.data_lambda = 1.0

        # --- 定义 IC & BC 配置 ---
        def is_on_big_circle(x):
            for circle in big_circles:
                if jnp.any(circle.on_boundary(x[:, :2])):
                    return True
            return False

        def is_on_small_circle(x):
            for circle in small_circles:
                if jnp.any(circle.on_boundary(x[:, :2])):
                    return True
            return False
        
        bc_config = [{
            'component': 0,
            'function': (lambda _: 0),
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _, u: 5 - u),
            'bc': (lambda x, on_boundary: jnp.any(on_boundary) and is_on_big_circle(x)),
            'type': 'robin'
        }, {
            'component': 0,
            'function': (lambda _, u: 1 - u),
            'bc': (lambda x, on_boundary: jnp.any(on_boundary) and is_on_small_circle(x)),
            'type': 'robin'
        }, {
            'component': 0,
            'function': (lambda _, u: 0.1 - u),
            'bc': (lambda x, on_boundary: jnp.any(on_boundary) and not is_on_big_circle(x) and not is_on_small_circle(x)),
            'type': 'robin'
        }]

        self.bcs = addbc(bc_config, self.geom_time)
        # --- pde data load ---
        def data_load(path):
            loader = DataLoader()
            loader.load(path,
                        input_dim=3,  # x, y, t
                        output_dim=1,  # u
                        t_transpose=True)
            data = loader.ref_data  # numpy (N, 3)
            X_all = jnp.array(data[:, :3], jnp.float32)  # shape (N, 3)
            y_all = jnp.array(data[:, 3:], jnp.float32)  # shape (N, 1)

            # # 根据定义域掩码对数据进行过滤
            # x_min, x_max = self.bbox[0], self.bbox[1]
            # y_min, y_max = self.bbox[2], self.bbox[3]
            # t_min, t_max = self.bbox[4], self.bbox[5]

            # mask = (
            #         (X_all[:, 0] >= x_min) & (X_all[:, 0] <= x_max) &
            #         (X_all[:, 1] >= y_min) & (X_all[:, 1] <= y_max) &
            #         (X_all[:, 2] >= t_min) & (X_all[:, 2] <= t_max)
            # )
            return X_all, y_all

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
            [self.bbox[0], self.bbox[1]],
            [self.bbox[2], self.bbox[3]],
            [self.bbox[4], self.bbox[5]],
        ]
        self.pde_sampler = LowDiscrepancySampler(self.X_pde, self.Y_pde, domain_bounds)

        if len(self.X_data) > self.BatchSize_data:
            self.is_batch = True
            self.data_size = self.BatchSize_data
            self.data_sampler = LowDiscrepancySampler(self.X_data, self.Y_data, domain_bounds)
        else:
            self.is_batch = False
            self.data_size = len(self.X_data)

        self.bcs_masks = [bc.filter(self.X_candidate) for bc in self.bcs]
        self.bcs_points = [
            X_ref[jnp.where(mask)[0]].reshape(-1, X_ref.shape[1]) if len(mask.shape) == 1 and mask.size > 0 else None
            for mask in self.bcs_masks
        ]
        valid_masks = [mask for mask in self.bcs_masks if mask is not None and len(mask.shape) == 1]
        max_shape = max((mask.shape[0] for mask in valid_masks), default=0)
        self.bcs_masks = [jnp.pad(mask, ((0, max_shape - mask.shape[0]),)) if mask is not None and len(mask.shape) == 1 else jnp.zeros(max_shape) for mask in self.bcs_masks]
        try:
            comb_mask = jnp.any(jnp.stack(self.bcs_masks, axis=1), axis=1)
        except ValueError as e:
            comb_mask = jnp.zeros_like(self.X_candidate[:, 0])  # 如果堆叠失败，返回一个零掩码作为安全措施

        # --- 定义 reset / step ---
        def reset_fn(key):
            X_eq, Y_eq = self.pde_sampler.get_batch(batch_size=self.BatchSize_eq)  # 采样内部可能存在循环问题
            if self.is_batch:
                X_d, Y_d = self.data_sampler.get_batch(batch_size=self.BatchSize_data)
            else:
                X_d, Y_d = self.X_data, self.Y_data
            masks = [bc.filter(X_eq) for bc in self.bcs]
            # 对 masks 做 shape 对齐（padding）
            valid_masks = [m for m in masks if m is not None and len(m.shape) == 1]
            max_shape = max((m.shape[0] for m in valid_masks), default=0)
            masks = [jnp.pad(m, ((0, max_shape - m.shape[0]),)) if m is not None and len(m.shape) == 1 else jnp.zeros(max_shape) for m in masks]
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
        u = pred[:, 0:1]
        u_t = pred[:, 1:2]  # 直接获取 u_t
        u_xx = pred[:, 2:3]  # 直接获取 u_xx
        u_yy = pred[:, 3:4]  # 直接获取 u_yy

        # 计算 PDE 残差：u_t = u_xx + u_yy
        r_u = u_t - u_xx - u_yy  # 计算残差

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
        r_all = self.pde_fn(pde_pred)  # shape: (N_pde, 2) -> [res_u, res_v]
        comb_mask = jnp.any(jnp.stack(bcs_masks, axis=1), axis=1)
        interior_mask = (~comb_mask).astype(r_all.dtype)
        r_masked = r_all * interior_mask[:, None]
        pde_loss = jnp.sum(r_masked ** 2) / (jnp.sum(interior_mask) + 1e-8)

        # ---- IC / BC loss ----
        bc_loss_sum, ic_loss_sum = 0.0, 0.0
        bc_count, ic_count = 0, 0

        for bc, mask in zip(self.bcs, bcs_masks):
            mask_f = mask[:, None].astype(pde_pred.dtype)
            err = bc.error(pde_pred, X_pde)  # shape: (N_pde, 1)
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

        jax.debug.print("pde_loss={:}, ic_loss={:}, bc_loss={:}, data_loss={:}", pde_loss, ic_loss, bc_loss, data_loss)

        return total_loss

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
train_task = Heat2D_ComplexGeometry()
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
