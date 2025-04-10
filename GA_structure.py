import datetime
import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *
import multiprocessing
from tqdm import trange



# ---- PARAMETERS ----
NUM_GENERATIONS = 100  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500
SCENARIO = 'Walker-v0'
POP = 50
MUTATION = 0.05
ELITE = 10
# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

CONTROLLER = alternating_gait

printS = []

def evaluate_fitness(robot_structure, view=False):    
    try:
        connectivity = get_full_connectivity(robot_structure)
  
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        t_reward = 0
        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        for t in range(STEPS):  
            # Update actuation before stepping
            actuation = CONTROLLER(action_size,t)
            if view:
                viewer.render('screen') 
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward

            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()
        return t_reward
    except (ValueError, IndexError) as e:
        return 0.0


def create_random_robot():
    """Generate a valid random robot structure."""
    
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot


def genetic_algorithm():
    try:
        global printS
        population = [create_random_robot() for _ in range(POP)]
        best_global = None
        best_fitness = -np.inf

        for gen in trange(NUM_GENERATIONS, desc='Evolving GA ', unit='gen'):
            with multiprocessing.Pool() as pool:
                fitness_values = pool.map(evaluate_fitness, population)
            # fitness_values = [evaluate_fitness(ind) for ind in population]
            current_best_idx = np.argmax(fitness_values)
            
            if fitness_values[current_best_idx] > best_fitness:
                best_fitness = fitness_values[current_best_idx]
                best_global = population[current_best_idx].copy()
            
            printS.append(f"Gen {gen+1}, Best: {best_fitness:.4f}, Avg: {np.mean(fitness_values):.4f}")

            elite_indices = np.argsort(fitness_values)[-ELITE:]
            elites = [population[i].copy() for i in elite_indices]

            new_population = elites.copy()
            
            while len(new_population) < POP:
                p1, p2 = tournament_selection(population, fitness_values)
                child = crossover(p1, p2)
                child = mutate(child, MUTATION)
                new_population.append(child)
            
            population = new_population

    except KeyboardInterrupt:
        printS.append("\n[INFO] Interrupted by user. Finalizing with current best found...")

    finally:
        return best_global, best_fitness

def crossover(parent1, parent2, max_attempts=10):

    for _ in range(max_attempts):
        mask = np.random.randint(0, 2, size=parent1.shape)
        child = np.where(mask, parent2, parent1)
        
        if is_connected(child) and has_actuator(child):
            return child
    
    return parent1 if evaluate_fitness(parent1) > evaluate_fitness(parent2) else parent2

def tournament_selection(population, fitness_values, k=5):
    selected_indices = random.sample(range(len(population)), k)
    selected = [(population[i], fitness_values[i]) for i in selected_indices]
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0], selected[1][0]

def mutate(individual, mutation_rate, max_attempts=5):
    original = individual.copy()
    
    for _ in range(max_attempts):
        mutated = individual.copy()
        flattened = mutated.flatten()
        num_mutations = int(len(flattened) * mutation_rate)
        
        for i in random.sample(range(len(flattened)), num_mutations):
            flattened[i] = (flattened[i] + random.randint(0, len(VOXEL_TYPES) - 1)) % len(VOXEL_TYPES)
        
        mutated = flattened.reshape(individual.shape)
        
        if is_connected(mutated) and has_actuator(mutated):
            return mutated
    
    return original


def run_ga():
    global printS
    best_robot, best_fitness = genetic_algorithm()
    if best_robot is not None:
        printS.append("Best robot structure found:")
        printS.append(best_robot)
        printS.append("Best fitness score:")
        printS.append(best_fitness)

        for i in range(5):
            utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
        now = datetime.now()
        timestamp = now.strftime("%d-%m")  # Format: day-month, e.g., "10-04"
        utils.create_gif(best_robot, filename=f'ga_{timestamp}.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)    
    else:
        printS.append("No valid robot was evolved.")

    for s in printS:
        print(s)   

if __name__ == "__main__":
    # Set the seed for reproducibility
    multiprocessing.freeze_support()

    for seed in utils.seed_list:
        utils.set_seed(utils.seed_list[0])
        run_ga()