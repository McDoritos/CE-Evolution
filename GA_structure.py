import datetime
import numpy as np
import random
import gymnasium as gym
from evogym.envs import *

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *
from tqdm import trange
from datetime import datetime
import os

base_dir = "gen_data_structures"
os.makedirs("gen_data_structures", exist_ok=True)


# ---- PARAMETERS ----
NUM_GENERATIONS = 50  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500
SCENARIO = 'Walker-v0'
POP = 50
MUTATION = 0.1
ELITE = 10
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


def genetic_algorithm(seed_folder):
    try:
        population = [create_random_robot() for _ in range(POP)]
        best_global = None
        best_fitness = -np.inf

        for gen in trange(NUM_GENERATIONS, desc='Evolving GA ', unit='gen'):
            #with multiprocessing.Pool() as pool:
            #    fitness_values = pool.map(evaluate_fitness, population)
            fitness_values = [evaluate_fitness(ind) for ind in population]

            population_with_fitness = list(zip(population, fitness_values))
            population_with_fitness = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)

            csv_filename = os.path.join(seed_folder, f"gen_{gen}.csv")
            utils.save_structure(population_with_fitness, csv_filename)

            if population_with_fitness[0][1] > best_fitness:
                best_fitness = population_with_fitness[0][1]
                best_global = population_with_fitness[0][0].copy()
            
            #print(f"Gen {gen+1}, Best: {best_fitness:.4f}, Avg: {np.mean(fitness_values):.4f}")
            
            print(f"Gen {gen+1}, Best: {best_fitness:.4f}, Avg: {np.mean(fitness_values):.4f}")

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
        #print("\n[INFO] Interrupted by user. Finalizing with current best found...")
        
        print("\n[INFO] Interrupted by user. Finalizing with current best found...")
        
        if 'population_with_fitness' in locals():
            utils.save_structures(gen+1, population_with_fitness, csv_filename)

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
        mutated = original.copy()
        # chooses random voxels for mutation
        mask = np.random.random(individual.shape) < mutation_rate
        if not np.any(mask):  # garants at least 1 mutation
            mask.flat[np.random.randint(0, individual.size)] = True
        
        for i, j in zip(*np.where(mask)):
            current = mutated[i, j]
            # choose a new voxel different from the current one
            new_val = random.choice([x for x in VOXEL_TYPES if x != current])
            mutated[i, j] = new_val
        
        if is_connected(mutated) and has_actuator(mutated):
            return mutated
    
    return original


def run_ga():

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ga_folder = os.path.join(base_dir, f"GA_Structure")
    os.makedirs(ga_folder, exist_ok=True)

    scenario_folder = os.path.join(ga_folder, SCENARIO)
    os.makedirs(scenario_folder, exist_ok=True)

    seed_folder = os.path.join(scenario_folder, f"seed_{seed} - {timestamp}")
    os.makedirs(seed_folder, exist_ok=True)

    best_robot, best_fitness = genetic_algorithm(seed_folder)
    if best_robot is not None:
        print("Best robot structure found:")
        print(best_robot)
        print("Best fitness score:")
        print(best_fitness)

        for i in range(5):
            utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
        
        utils.create_gif(best_robot, filename=f'ga_{timestamp}.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)    
        utils.save_plot(seed_folder, SCENARIO, seed)
    else:
        print("No valid robot was evolved.")


if __name__ == "__main__":
    for scenario in utils.scenarios_3_1:
        SCENARIO = scenario
        for seed in utils.seed_list:
            utils.set_seed(seed)
            run_ga()