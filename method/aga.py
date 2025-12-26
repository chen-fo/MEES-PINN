import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
import jax
import jax.numpy as jnp
from src.pde.Burgers1D import get_fitness, policy
import time
import jax
import jax.numpy as jnp

def aga(get_fitness, n=50, m=260, elite_rate=0.05, seed=1, max_seconds=60.0, lower=-10, upper=10.0):
    key = jax.random.PRNGKey(seed)
    key, k1 = jax.random.split(key)
    pop = jax.random.uniform(k1, shape=(n, m), minval=lower, maxval=upper)

    elite = max(1, int(n * elite_rate))

    cx_min, cx_max = 0.60, 0.95
    pm_min, pm_max = 0.001, 0.02     # ~ 1/m ≈ 0.0038
    t_min,  t_max  = 2, 4            # 锦标赛规模

    mut_sigma = 0.10 * (upper - lower)
    no_improve = 0
    it = 0
    nt = 0

    fitness = get_fitness(pop)
    best_idx = int(jnp.argmax(fitness))
    best_fit = float(fitness[best_idx])
    best_x = pop[best_idx]

    while nt < max_seconds:
        it += 1
        st = time.time()
        std = jnp.std(pop, axis=0)
        D = jnp.clip(std / (upper - lower + 1e-9), 0.0, 1.0).mean()

        cx_rate   = float(cx_min + (cx_max - cx_min) * (1.0 - D))
        pm_base   = float(pm_min + (pm_max - pm_min) * (1.0 - D))
        t = int(t_min + jnp.floor((1.0 - D) * (t_max - t_min)))
        t = max(t_min, min(t, t_max))

        if no_improve >= 10:
            pm_base = min(pm_base * 1.5, pm_max)
            no_improve = 0

        elite_idx = jnp.argsort(-fitness)[:elite]   
        elites = pop[elite_idx]

        def tournament_select(k, pop, fitness, ksize):
            idx = jax.random.randint(k, (ksize,), 0, pop.shape[0])
            return idx[jnp.argmax(fitness[idx])]

        children = []
        parent_fits = [] 
        num_children = n - elite
        for _ in range(num_children):
            key, k1 = jax.random.split(key)
            key, k2 = jax.random.split(key)
            p1 = tournament_select(k1, pop, fitness, t)
            p2 = tournament_select(k2, pop, fitness, t)

            key, k3 = jax.random.split(key)
            key, k4 = jax.random.split(key)
            do_cx = jax.random.bernoulli(k3, cx_rate)
            mask = jax.random.bernoulli(k4, 0.5, shape=(m,))
            child = jnp.where(do_cx & mask, pop[p1], pop[p2])

            children.append(child)
            parent_fits.append((fitness[p1] + fitness[p2]) / 2.0)

        children = jnp.stack(children, axis=0)  # (n-elite, m)
        parent_fits = jnp.stack(parent_fits, axis=0) 

        f_avg = float(jnp.mean(fitness))
        f_max = float(jnp.max(fitness))
        denom = (f_max - f_avg + 1e-8)
        pm_i = jnp.where(
            parent_fits >= f_avg,
            pm_min + (pm_max - pm_min) * (f_max - parent_fits) / denom,
            pm_max,
        )
        pm_i = jnp.maximum(pm_i, pm_base)  # (n-elite,)

        key, km1 = jax.random.split(key)
        key, km2 = jax.random.split(key)
        mut_mask = jax.random.bernoulli(km1, pm_i[:, None], shape=children.shape)
        noise = jax.random.normal(km2, shape=children.shape) * mut_sigma
        children = jnp.clip(children + mut_mask * noise, lower, upper)

        pop = jnp.vstack([elites, children])

        fitness = get_fitness(pop)
        curr_idx = int(jnp.argmax(fitness))
        curr_fit = float(fitness[curr_idx])
        if curr_fit > best_fit + 1e-12:
            best_fit = curr_fit
            best_x = pop[curr_idx]
            no_improve = 0
        else:
            no_improve += 1
        nt += time.time() - st
        print(f"time:{nt:.2f} iter {it}: best fitness = {best_fit:.8f}")

    return best_x, best_fit


if __name__ == '__main__':
    n = 50
    m = policy.num_params
    # def f(params):
    #     seed = int(time.time())  
    #     key = jax.random.PRNGKey(seed)

    #     arr = jax.random.normal(key, shape=(n,))   
    #     return arr
    
    aga(get_fitness, n, m)
