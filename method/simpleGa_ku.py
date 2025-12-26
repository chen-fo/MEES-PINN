import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def crossover(parent1, parent2, eta=20, key=None):
    u = jax.random.uniform(key, shape=parent1.shape)
    
    beta = jnp.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    
    child1 = 0.5 * ((parent1 + parent2) - beta * (parent2 - parent1))
    child2 = 0.5 * ((parent1 + parent2) + beta * (parent2 - parent1))
    
    return child1, child2
@jax.jit
def gSbx(p1, p2, g1, g2, eta=20, key=None):
    keys = jax.random.split(key, 8)
    shape = p1.shape

    
    
    u1 = jnp.where(
        (g1 * (p1 - p2) > 0),   
        jax.random.uniform(keys[0], shape=shape, minval=0, maxval=0.5), 
        jnp.where(
            (g2 * (p1 - p2) < 0),
            jax.random.uniform(keys[1], shape=shape, minval=0.5, maxval=1), 
            jax.random.uniform(keys[2], shape=shape, minval=0, maxval=1)  
        )
    )
    u2 = jnp.where(
        (g2 * (p1 - p2) < 0),   
        jax.random.uniform(keys[3], shape=shape, minval=0, maxval=0.5), 
        jnp.where(
            (g2 * (p1 - p2) > 0),
            jax.random.uniform(keys[4], shape=shape, minval=0.5, maxval=1), 
            jax.random.uniform(keys[5], shape=shape, minval=0, maxval=1)  
        )
    )
    beta1 = jnp.where(u1 <= 0.5, (2 * u1) ** (1 / (eta + 1)), (1 / (2 * (1 - u1))) ** (1 / (eta + 1)))
    beta2 = jnp.where(u2 <= 0.5, (2 * u2) ** (1 / (eta + 1)), (1 / (2 * (1 - u2))) ** (1 / (eta + 1)))
    
    lambda_1 = jax.random.uniform(keys[6], shape=shape)
    lambda_2 = jax.random.uniform(keys[7], shape=shape)
    a1 = jnp.where((lambda_1 <= 0.5) & (g1 * (p1 - p2) > 0) & (g2 * (p1 - p2) < 0), 
    -beta2 - 1,
    beta1 - 1)
    a2 = jnp.where((lambda_2 <= 0.5) & (g1 * (p1 - p2) > 0) & (g2 * (p1 - p2) < 0),
    beta1 - 1,
    -beta2 - 1)
    '''repair'''
    # a1 = jnp.where(p1 == 0, 0, a1)
    # a2 = jnp.where(p2 == 0, 0, a2)

    c1 = p1 + a1 * (p1 - p2) / 2
    c2 = p2 - a2 * (p1 - p2) / 2
    return c1, c2
@jax.jit
def mutation(individual, eta_m=20, key=None):
    u = jax.random.uniform(key, shape=individual.shape)

    delta = jnp.where(u < 0.5, 
                      (2 * u) ** (1 / (eta_m + 1)) - 1, 
                      1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
    
    mutated_individual = individual + delta
    
    return mutated_individual
@partial(jax.jit, static_argnums=(2,))
def elitism(population, fitness, elite_size=1):
    elite_indices = jnp.argsort(fitness)[-elite_size:]
    elite_individuals = population[elite_indices]
    elite_fitness = fitness[elite_indices]
    
    return elite_individuals, elite_fitness
def initialize_population(pop_size, dim, dix):
    population =  jax.random.normal(jax.random.PRNGKey(dix), shape=(pop_size, dim))
    return population
@jax.jit
def selection(population, scores, key):
    probabilities = 1 / jnp.abs(scores)
    total_probability = jnp.sum(probabilities)
    probabilities = probabilities / total_probability
    cumulative_probabilities = jnp.cumsum(probabilities)
    random_values = jax.random.uniform(key, shape=(2,))
    selected_indices = jnp.searchsorted(cumulative_probabilities, random_values)
    selected_individuals = population[selected_indices]
    
    return selected_individuals[0], selected_individuals[1]
