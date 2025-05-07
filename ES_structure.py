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

base_dir = "gen_data_structures"
os.makedirs(base_dir, exist_ok=True)

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
        print(f"[ERROR] Invalid robot structure: {robot_structure}. Error: {e}")
        return 0.0

def create_random_robot():
    """Generate a valid random robot structure."""
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot

def evolutionary_strategy(seed_folder):
    try:
        population = [create_random_robot() for _ in range(POPULATION_SIZE)]
        best_global = None
        best_fitness = -np.inf

        for gen in trange(NUM_GENERATIONS, desc=f"ES Structure {seed}", unit="gen"):
            csv_filename = os.path.join(seed_folder, f"gen_{gen}.csv")
            offsprings = [mutate(random.choice(population)) for _ in range(OFFSPRING_SIZE)]
            combined_population = population + offsprings

            fitness_values = [evaluate_fitness(individual) for individual in combined_population]

            population_with_fitness = list(zip(combined_population, fitness_values))
            population_with_fitness = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
            
            try:
                utils.save_structure(population_with_fitness, csv_filename)
            except :
                print("Error writing to CSV file. Skipping save.")
            if population_with_fitness[0][1] > best_fitness:
                best_fitness = population_with_fitness[0][1]
                best_global = population_with_fitness[0][0].copy()
            
            avg_fitness = sum(fitness for _, fitness in population_with_fitness) / len(population_with_fitness)

            population = [ind for ind, _ in population_with_fitness[:POPULATION_SIZE]]

            print(f"Gen {gen+1}, Best: {best_fitness:.4f}, Avg: {avg_fitness:.4f}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Saving current state...")
        if 'population_with_fitness' in locals():
            utils.save_structure(gen+1, population_with_fitness, csv_filename)
    except RuntimeError as e:
        print(e)
    finally:
        return best_global, best_fitness


def mutate(parent, max_attempts=5):
    shape = parent.shape
    total_cells = parent.size
    num_mutations = int(total_cells * MUTATION)

    for _ in range(max_attempts):
        child = copy.deepcopy(parent)
        child = child.flatten()
        indices = random.sample(range(total_cells), num_mutations)

        for idx in indices:
            current_value = child[idx]
            choices = [v for v in VOXEL_TYPES if v != current_value]
            child[idx] = random.choice(choices)

        child = child.reshape(shape)

        if is_connected(child) and has_actuator(child):
            return child

    #print(f"[WARNING] Mutation failed after {max_attempts} attempts. Returning original parent.")
    return parent  # Return the original parent if no valid mutation is found


def run_es(seed):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    es_folder = os.path.join(base_dir, f"ES_Structure")
    os.makedirs(es_folder, exist_ok=True)

    scenario_folder = os.path.join(es_folder, SCENARIO)
    os.makedirs(scenario_folder, exist_ok=True)

    seed_folder = os.path.join(scenario_folder, f"seed_{seed} - {timestamp}")
    os.makedirs(seed_folder, exist_ok=True)
    
    best_robot, best_fitness = evolutionary_strategy(seed_folder)
    if best_robot is not None:
        print("Best robot structure found:")
        print(best_robot)
        print("Best fitness score:")
        print(best_fitness)
        

        utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)



        gif_filename = os.path.join(seed_folder, f"_best.gif")
        utils.create_gif(best_robot, filename=gif_filename, scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)                                                                                 
    else:
        print("No valid robot was evolved.")

if __name__ == "__main__":
    for scenario in utils.scenarios:
        SCENARIO = scenario
        for seed in utils.seed_list:
            print(f"Running ES with scenario: {SCENARIO} and seed: {seed}")
            utils.set_seed(seed)
            run_es(seed)

