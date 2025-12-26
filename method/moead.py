import jax
import jax.numpy as jnp
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from src.pde.Burgers1D import get_fitness, policy

def sbx_crossover(key, p1: jnp.ndarray, p2: jnp.ndarray,
                  pc=0.9, eta_c=15.0, lower=0.0, upper=1.0):

    B, L = p1.shape
    key, kprob, ku = jax.random.split(key, 3)
    do_cx = jax.random.bernoulli(kprob, pc, (B, 1))
    u = jax.random.uniform(ku, (B, L), minval=0.0, maxval=1.0)

    beta = jnp.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta_c + 1.0)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
    )
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    c1 = jnp.where(do_cx, c1, p1)
    c2 = jnp.where(do_cx, c2, p2)

    c1 = jnp.clip(c1, lower, upper)
    c2 = jnp.clip(c2, lower, upper)
    return key, c1, c2

def polynomial_mutation(key, X: jnp.ndarray, pm=0.01, eta_m=20.0,
                        lower=0.0, upper=1.0):
    N, L = X.shape
    key, kmask, kr = jax.random.split(key, 3)
    mask = jax.random.bernoulli(kmask, pm, X.shape)
    r = jax.random.uniform(kr, X.shape, minval=0.0, maxval=1.0)

    xl = jnp.asarray(lower)
    xu = jnp.asarray(upper)
    # 广播成 (1,L) 再到 (N,L)
    if xl.ndim == 0:
        xl = jnp.full((1, L), xl)
    elif xl.ndim == 1:
        xl = jnp.broadcast_to(xl[None, :], (1, L))
    if xu.ndim == 0:
        xu = jnp.full((1, L), xu)
    elif xu.ndim == 1:
        xu = jnp.broadcast_to(xu[None, :], (1, L))

    width = xu - xl + 1e-12
    delta1 = (X - xl) / width
    delta2 = (xu - X) / width
    mut_pow = 1.0 / (eta_m + 1.0)

    r_less = (r < 0.5)

    xy1 = 1.0 - delta1
    val1 = 2.0 * r + (1.0 - 2.0 * r) * (xy1 ** (eta_m + 1.0))
    deltaq1 = (val1 ** mut_pow) - 1.0

    xy2 = 1.0 - delta2
    val2 = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy2 ** (eta_m + 1.0))
    deltaq2 = 1.0 - (val2 ** mut_pow)

    deltaq = jnp.where(r_less, deltaq1, deltaq2)
    X_mut = X + jnp.where(mask, deltaq * width, 0.0)
    X_mut = jnp.clip(X_mut, xl, xu)
    return key, X_mut



def _tchebycheff(F: jnp.ndarray, lamb: jnp.ndarray, z: jnp.ndarray):
    return jnp.max(lamb[None, :] * jnp.abs(F - z[None, :]), axis=-1)


def moead(get_fitness,
          n=50, m=260, generations=2000, seed=0,
          lower=0.0, upper=1.0,
          pc=0.9, pm=None, eta_c=15.0, eta_m=20.0,
          T=None, delta=0.9, nr=2):

    if pm is None:
        pm = 1.0 / m
    if T is None:
        T = max(5, int(0.1 * n))

    key = jax.random.PRNGKey(seed)

    key, k0 = jax.random.split(key)
    pop = jax.random.uniform(k0, (n, m), minval=lower, maxval=upper)

    objs = get_fitness(pop)
    k = int(objs.shape[1])

    key, kw = jax.random.split(key)
    lambdas = jax.random.dirichlet(kw, jnp.ones((k,)), (n,))  # (n,k)

    dists = jnp.linalg.norm(lambdas[:, None, :] - lambdas[None, :, :], axis=-1)  # (n,n)
    neighbors = jnp.argsort(dists, axis=1)[:, :T]  # (n,T)

    z = jnp.min(objs, axis=0)

    for gen in range(1, generations + 1):
        key, kperm = jax.random.split(key)
        order = jax.random.permutation(kperm, n)

        for i in order.tolist():
            Bi = neighbors[i]  # (T,)

            key, kprob = jax.random.split(key)
            use_nb = jax.random.bernoulli(kprob, delta)

            if use_nb:
                key, ksel = jax.random.split(key)
                parents = jax.random.choice(ksel, Bi, shape=(2,), replace=False)
            else:
                key, ksel = jax.random.split(key)
                parents = jax.random.randint(ksel, (2,), 0, n)

            p1 = pop[parents[0]][None, :]  # (1,m)
            p2 = pop[parents[1]][None, :]  # (1,m)

            key, c1, c2 = sbx_crossover(key, p1, p2, pc=pc, eta_c=eta_c, lower=lower, upper=upper)
            key, kpick = jax.random.split(key)
            pick_c1 = jax.random.bernoulli(kpick, 0.5)
            child = jnp.where(pick_c1, c1, c2)  # (1,m)
            key, child = polynomial_mutation(key, child, pm=pm, eta_m=eta_m, lower=lower, upper=upper)
            child = child[0]  # (m,)

            child_obj = get_fitness(child[None, :])[0]  # (k,)

      
            z = jnp.minimum(z, child_obj)
  
            key, knei = jax.random.split(key)
            Bi_perm = jax.random.permutation(knei, Bi.shape[0])
            replaced = 0
            for jj in Bi[Bi_perm].tolist():
               
                f_old = _tchebycheff(objs[jj:jj+1, :], lambdas[jj], z)[0]
                f_new = _tchebycheff(child_obj[None, :], lambdas[jj], z)[0]
                if f_new <= f_old:
                    pop = pop.at[jj].set(child)
                    objs = objs.at[jj].set(child_obj)
                    replaced += 1
                    if replaced >= nr:
                        break

        sum_losses = jnp.sum(objs, axis=1)
        best_idx = int(jnp.argmin(sum_losses))
        best_losses = objs[best_idx]
        best_total = float(jnp.sum(best_losses))
        best_losses_list = [float(v) for v in best_losses.tolist()]
        print(f"iter {gen}: best_total={best_total:.6f}, losses={best_losses_list}")

    return pop, objs, lambdas, neighbors
if __name__ == '__main__':
    n = 50
    m = policy.num_params
    # def f(params):
    #     seed = int(time.time())  
    #     key = jax.random.PRNGKey(seed)

    #     arr = jax.random.normal(key, shape=(n,))   
    #     return arr
    
    moead(get_fitness, n, m)