import copy
import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, sample_robot, get_full_connectivity, is_connected

from tqdm import trange
import utils
from fixed_controllers import *
from datetime import datetime

os.makedirs("gen_data_structures", exist_ok=True)

# ---- PARAMETERS ----
NUM_GENERATIONS = 100  # Number of generations to evolve
STEPS = 500
SCENARIO = 'Walker-v0'

MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
MUTATION = 0.1
POPULATION_SIZE = 10
OFFSPRING_SIZE = 20
# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)
CONTROLLER = alternating_gait

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
        population = [create_random_robot() for _ in range(POPULATION_SIZE)]
        best_global = None
        best_fitness = -np.inf

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join("gen_data_structures", f"ES-Structure_evolution-data_{timestamp}.csv")

        for gen in trange(NUM_GENERATIONS, desc="Evolving ES Structure", unit="gen"):
            offsprings = [mutate(random.choice(population)) for _ in range(OFFSPRING_SIZE)]
            combined_population = population + offsprings

            fitness_values = [evaluate_fitness(individual) for individual in combined_population]

            population_with_fitness = list(zip(combined_population, fitness_values))
            population_with_fitness = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
            
            utils.save_structures(gen+1, population_with_fitness, csv_filename)

            if population_with_fitness[0][1] > best_fitness:
                best_fitness = population_with_fitness[0][1]
                best_global = population_with_fitness[0][0].copy()
            
            avg_fitness = sum(fitness for _, fitness in population_with_fitness) / len(population_with_fitness)

            population = [ind for ind, _ in population_with_fitness[:POPULATION_SIZE]]

            print(f"Gen {gen+1}, Best: {best_fitness:.4f}, Avg: {avg_fitness:.4f}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Saving current state...")
        if 'population_with_fitness' in locals():
            utils.save_structures(gen+1, population_with_fitness, csv_filename)
    finally:
        return best_global, best_fitness


def mutate(parent, max_attempts=5):
    child = copy.deepcopy(parent)
    for _ in range(max_attempts):
        shape = child.shape

        child = child.flatten()
        
        total_cells = child.size
        num_mutations = int(total_cells * MUTATION)

        indices = random.sample(range(total_cells), num_mutations)
        for idx in indices:
            current_value = child[idx]
            choices = [v for v in VOXEL_TYPES if v != current_value]
            child[idx] = random.choice(choices)

        child = child.reshape(shape)

        if is_connected(child) and has_actuator(child):
            return child

    return child


def run_es():
    best_robot, best_fitness = evolutionary_strategy()
    if best_robot is not None:
        print("Best robot structure found:")
        print(best_robot)
        print("Best fitness score:")
        print(best_fitness)

        for i in range(5):
            utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
        now = datetime.now()
        timestamp = now.strftime("%d-%m")  # Format: day-month, e.g., "10-04"
        utils.create_gif(best_robot, filename=f'es_{timestamp}.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)                                                                                 
    else:
        print("No valid robot was evolved.")

if __name__ == "__main__":
    for seed in utils.seed_list:
        utils.set_seed(seed)
        run_es()