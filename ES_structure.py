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
MUTATION = 0.05
MU = 2
LAMBDA = 10
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

def evolutionary_strategy():
    try:
        global printS
        population = [create_random_robot() for _ in range(MU)]
        best_global = None
        best_fitness = -np.inf

        for gen in trange(NUM_GENERATIONS, desc="Evolving ES ", unit="gen"):
            offsprings = [mutate(population[random.randint(0, len(population) - 1)]) for _ in range(LAMBDA)]
            population.extend(offsprings)

                        # Parallel fitness evaluation
            with multiprocessing.Pool() as pool:
                fitness_values = pool.map(evaluate_fitness, population)

            # Pair population with fitness
            population_with_fitness = list(zip(population, fitness_values))
            population_with_fitness = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
            
            if population_with_fitness[0][1] > best_fitness:
                best_fitness = population_with_fitness[0][1]
                best_global = population_with_fitness[0][0].copy()
            
            # Calculate the mean fitness value without extracting the list
            avg_fitness = sum(fitness for _, fitness in population_with_fitness) / len(population_with_fitness)
            printS.append(f"Gen {gen+1}, Best: {best_fitness:.4f}, Avg: {avg_fitness:.4f}")

            population = [ind for ind, _ in population_with_fitness[:MU]]

    except KeyboardInterrupt:
        printS +=("\n[INFO] Interrupted by user. Finalizing with current best found...")

    finally:
        return best_global, best_fitness
    



def mutate(parent, max_attempts=5):
    original = parent.copy()
    
    for _ in range(max_attempts):
        mutated = parent.copy()
        flattened = mutated.flatten()
        num_mutations = int(len(flattened) * MUTATION)
        
        for i in random.sample(range(len(flattened)), num_mutations):
            flattened[i] = (flattened[i] + random.randint(0, len(VOXEL_TYPES) - 1)) % len(VOXEL_TYPES)
        
        mutated = flattened.reshape(original.shape)
        
        if is_connected(mutated) and has_actuator(mutated):
            return mutated
    
    return original


def run_es():
    global printS
    best_robot, best_fitness = evolutionary_strategy()
    if best_robot is not None:
        printS +=("Best robot structure found:")
        printS +=(best_robot)
        printS +=("Best fitness score:")
        printS +=(best_fitness)

        for i in range(5):
            utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
        now = datetime.now()
        timestamp = now.strftime("%d-%m")  # Format: day-month, e.g., "10-04"
        utils.create_gif(best_robot, filename=f'es_{timestamp}.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)                                                                                 
    else:
        printS +=("No valid robot was evolved.")
    
    for s in printS:
        print(s)  

if __name__ == "__main__":

    multiprocessing.freeze_support()

    for seed in utils.seed_list:
        utils.set_seed(utils.seed_list[0])
        run_es()