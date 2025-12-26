"""
CMA-ES implementation without JAX JIT compilation.
This version removes JIT compilation to allow for performance comparison with the JIT-accelerated version.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from evojax.algo import CMA_ES_JAX
from flax.struct import dataclass
from typing import List


@dataclass
class Result:
    best_w: jnp.ndarray
    best_fit: float
    evals: int
    iter_time_ls: List[float]
    loss_ls: List[float]
    various_loss_ls: List[float]


def train(get_fitness, policy, sim_mgr, pop_size=50, init_stdev=0.02, max_iters=5000, seed=0):
    # 创建一个修改版的CMA_ES_JAX类，禁用jit
    class CMA_ES_JAX_No_JIT(CMA_ES_JAX):
        def __init__(self, *args, **kwargs):
            # 先调用父类初始化
            super().__init__(*args, **kwargs)
            
        # 重写_eigen_decomposition方法，移除jit
        def _eigen_decomposition(self):
            if self.state.B is not None and self.state.D is not None:
                return self.state.B, self.state.D
            # 直接调用未装饰的函数
            B, D, C = _eigen_decomposition_core_no_jit(self.state.C)
            self.state = self.state._replace(B=B, D=D, C=C)
            return B, D
            
        # 重写_ask方法，移除jit
        def _ask(self, n_samples):
            """Real implementaiton of ask, which samples multiple parameters in parallel."""
            n_dim = self.hyper_parameters.n_dim
            mean, sigma = self.state.mean, self.state.sigma
            B, D = self._eigen_decomposition()

            key, subkey = jax.random.split(self.state.key)
            self.state = self.state._replace(key=key)
            subkey = jax.random.split(subkey, n_samples)
            x = _batch_sample_solution_no_jit(subkey, B, D, n_dim, mean, sigma)

            return x
            
        # 重写tell方法，移除jit
        def tell(self, fitness: jnp.ndarray, solutions: jnp.ndarray = None) -> None:
            """Tell evaluation values as fitness."""
            if solutions is None:
                assert self._latest_solutions is not None, \
                    "`soltuions` is not given, expecting using latest samples but this was not done."
                assert self._latest_solutions.shape[0] == self.hyper_parameters.pop_size, \
                    f"Latest samples (shape={self._latest_solutions.shape}) not having pop_size-length ({self.hyper_parameters.pop_size})."
                solutions = self._latest_solutions
            else:
                assert solutions.shape[0] == self.hyper_parameters.pop_size, \
                    "Given solutions must have pop_size-length, which is not true."
                
            # We want maximization, while the following logics is for minimimzation.
            # Handle this calse by simply revert fitness
            fitness = - fitness

            # real computation
            # - must do it as _tell_core below expects B, C, D to be computed.
            B, D = self._eigen_decomposition()
            self.state = self.state._replace(B=B, D=D)

            # 调用未装饰的函数
            next_state = _tell_core_no_jit(hps=self.hyper_parameters, coeff=self.coefficients,
                                          state=self.state, fitness=fitness, solutions=solutions)

            self.state = next_state

    # 定义无jit版本的辅助函数
    def _eigen_decomposition_core_no_jit(_C):
        _C = (_C + _C.T) / 2
        D2, B = jnp.linalg.eigh(_C)
        D = jnp.sqrt(jnp.where(D2 < 0, 1e-8, D2))
        _C = jnp.dot(jnp.dot(B, jnp.diag(D ** 2)), B.T)
        _B, _D = B, D
        return _B, _D, _C
        
    def _sample_solution_no_jit(key, B, D, n_dim, mean, sigma) -> jnp.ndarray:
        z = jax.random.normal(key, shape=(n_dim,))   # ~ N(0, I)
        y = B.dot(jnp.diag(D)).dot(z)  # ~ N(0, C)
        x = mean + sigma * y  # ~ N(m, σ^2 C)
        return x

    def _batch_sample_solution_no_jit(subkey, B, D, n_dim, mean, sigma):
        # 移除了jax.vmap和jax.jit装饰器，改为循环实现
        batch_size = subkey.shape[0]
        result = []
        for i in range(batch_size):
            sample = _sample_solution_no_jit(subkey[i], B, D, n_dim, mean, sigma)
            result.append(sample)
        return jnp.array(result)

    def _tell_core_no_jit(hps, coeff, state, fitness, solutions):
        next_state = state

        g = state.g + 1
        next_state = next_state._replace(g=g)

        ranking = jnp.argsort(fitness, axis=0)

        sorted_solutions = solutions[ranking]

        B, D = state.B, state.D

        # Sample new population of search_points, for k=1, ..., pop_size
        B, D = state.B, state.D  # already computed.
        next_state = next_state._replace(B=None, D=None)

        x_k = jnp.array(sorted_solutions)  # ~ N(m, σ^2 C)
        y_k = (x_k - state.mean) / state.sigma  # ~ N(0, C)

        # Selection and recombination
        # use lax.dynamic_slice_in_dim here:
        y_w = jnp.sum(
            jax.lax.dynamic_slice_in_dim(y_k, 0, hps.mu, axis=0).T *
            jax.lax.dynamic_slice_in_dim(coeff.weights,  0, hps.mu, axis=0),
            axis=1,
        )
        mean = state.mean + coeff.cm * state.sigma * y_w
        next_state = next_state._replace(mean=mean)

        # Step-size control
        C_2 = B.dot(jnp.diag(1 / D)).dot(B.T)  # C^(-1/2) = B D^(-1) B^T
        p_sigma = (1 - coeff.c_sigma) * state.p_sigma + jnp.sqrt(
            coeff.c_sigma * (2 - coeff.c_sigma) * coeff.mu_eff
        ) * C_2.dot(y_w)
        next_state = next_state._replace(p_sigma=p_sigma)

        norm_p_sigma = jnp.linalg.norm(state.p_sigma)
        sigma = state.sigma * jnp.exp(
            (coeff.c_sigma / coeff.d_sigma) * (norm_p_sigma / coeff.chi_n - 1)
        )
        sigma = jnp.min(jnp.array([sigma, 1e32]))
        next_state = next_state._replace(sigma=sigma)

        # Covariance matrix adaption
        h_sigma_cond_left = norm_p_sigma / jnp.sqrt(
            1 - (1 - coeff.c_sigma) ** (2 * (state.g + 1))
        )
        h_sigma_cond_right = (1.4 + 2 / (hps.n_dim + 1)) * coeff.chi_n
        # h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0  # (p.28)
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0

        # (eq.45)
        pc = (1 - coeff.cc) * state.pc + h_sigma * \
            jnp.sqrt(coeff.cc * (2 - coeff.cc) * coeff.mu_eff) * y_w
        next_state = next_state._replace(pc=pc)

        # (eq.46)
        w_io = coeff.weights * jnp.where(
            coeff.weights >= 0,
            1,
            hps.n_dim / (jnp.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + 1e-8),
        )

        delta_h_sigma = (1 - h_sigma) * coeff.cc * (2 - coeff.cc)  # (p.28)
        # assert delta_h_sigma <= 1

        # (eq.47)
        rank_one = jnp.outer(state.pc, state.pc)

        # 使用循环替代可能导致OOM的操作
        rank_mu = jnp.zeros_like(state.C)
        for w, y in zip(w_io, y_k):
            rank_mu = rank_mu + w * jnp.outer(y, y)

        C = (
            (
                1
                + coeff.c1 * delta_h_sigma
                - coeff.c1
                - coeff.cmu * jnp.sum(coeff.weights)
            )
            * state.C
            + coeff.c1 * rank_one
            + coeff.cmu * rank_mu
        )
        next_state = next_state._replace(C=C)

        return next_state

    # 创建solver实例
    solver = CMA_ES_JAX_No_JIT(
        pop_size=pop_size,
        init_stdev=init_stdev,
        param_size=policy.num_params,
        seed=seed,
    )

    loss_ls = []
    various_loss_ls = []
    iter_time_ls = []
    runtime = 0.0
    train_iters = 0

    best_loss = np.inf
    best_flat_params = None

    while train_iters < max_iters:
        t0 = time.time()
        params = solver.ask()
        losses, scores = get_fitness(sim_mgr, params)
        solver.tell(fitness=scores)

        avg_loss = np.mean(np.array(scores, copy=False))
        various_loss = np.mean(np.array(losses, copy=False), axis=0)

        loss_ls.append(-avg_loss)
        various_loss_ls.append(various_loss)

        idx_best = int(np.argmax(scores))
        cur_best_loss = float(-scores[idx_best])  # fitness = -loss

        if cur_best_loss < best_loss:
            best_loss = cur_best_loss
            best_flat_params = np.array(params[idx_best], copy=True)

        elapsed = time.time() - t0
        iter_time_ls.append(elapsed)
        runtime += elapsed
        train_iters += 1

        print(f"iter={train_iters:5d}  time={runtime:6.2f}s  loss(avg)={loss_ls[-1]:.2e}  pde_loss={various_loss[0]:.2e} ic_loss={various_loss[1]:.2e} bc_loss={various_loss[2]:.2e} data_loss={various_loss[3]:.2e}")

    print(f"\nFinished at iter={train_iters}, last loss(avg)={loss_ls[-1]:.2e}, best loss={best_loss:.2e}")

    return Result(best_w=best_flat_params, best_fit=best_loss, evals=max_iters, iter_time_ls=iter_time_ls, loss_ls=loss_ls, various_loss_ls=various_loss_ls)