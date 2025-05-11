import imageio
import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *

from tqdm import trange
import utils
from datetime import datetime
import multiprocessing

base_dir = "gen_data_controllers"
os.makedirs("gen_data_controllers", exist_ok=True)


NUM_GENERATIONS = 100  # Number of generations to evolve
STEPS = 500
SEED = 42
POPULATION_SIZE = 9
OFFSPRING_SIZE = 16


robot_structure = np.array([ 
[1,3,1,0,0],
[4,1,3,2,2],
[3,4,4,4,4],
[3,0,0,3,2],
[0,0,0,0,2]
])




# ---- FITNESS FUNCTION ----
def evaluate_fitness(weights, scenario, connectivity, brain, view=False):
        set_weights(brain, weights)  # Load weights into the network
        env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        sim = env
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        state = env.reset()[0]  # Get initial state
        t_reward = 0
        for t in range(STEPS):  
            # Update actuation before stepping
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
            action = brain(state_tensor).detach().numpy().flatten() # Get action
            if view:
                viewer.render('screen') 
            state, reward, terminated, truncated, info = env.step(action)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()
        return t_reward 

def mutate(individual, noise_std=0.1):
    """Mutate the weights by adding Gaussian noise."""
    mutated = []
    for param in individual:
        noise = np.random.normal(0, noise_std, param.shape)
        mutated.append(param + noise)
    return mutated

# ---- EVOLUTIONARY STRATEGY ALGORITHM ----
def es(seed_folder, brain, scenario, connectivity):
    try:
        # 1 - Generate population
        population = [ [np.random.randn(*param.shape) for param in brain.parameters()] for _ in range(POPULATION_SIZE) ]
        best_individual = None
        best_individual_reward = -np.inf

        best_rewards = []
        mean_rewards = []

        for gen in trange(NUM_GENERATIONS, desc="Evolving ES", unit="gen"):

            # 2 - Generate offspring
            offspring = [ mutate(random.choice(population)) for _ in range(OFFSPRING_SIZE) ]
            combined_population = population + offspring

            fitness_values = [evaluate_fitness(individual, scenario, connectivity, brain) for individual in combined_population]

            population_with_fitness = list(zip(combined_population, fitness_values))
            population_with_fitness = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
            
            csv_filename = os.path.join(seed_folder, f"gen_{gen}.csv")
            utils.save_controller(population_with_fitness, csv_filename)

            # Selecting best individual from the population
            if population_with_fitness[0][1] > best_individual_reward:
                best_individual_reward = population_with_fitness[0][1]
                best_individual = population_with_fitness[0][0].copy()
            
            # Getting metrics
            avg_fitness = sum(fitness for _, fitness in population_with_fitness) / len(population_with_fitness)
            best_rewards.append(best_individual_reward)
            mean_rewards.append(avg_fitness)

            # 3 - Keeping best individuals
            population = [ind for ind, _ in population_with_fitness[:POPULATION_SIZE]]

            print(f"Generation {gen + 1}: Best Reward = {best_individual_reward:.4f}, Mean Reward = {avg_fitness:.4f}")

        print("Best Fitness Achieved: ", best_individual_reward)
        set_weights(brain, best_individual)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Saving current state...")
        if 'population_with_fitness' in locals():
            utils.save_controllers(gen+1, population_with_fitness, csv_filename)
        
        if best_individual is not None:
            print(f"Best reward achieved: {best_individual_reward:.4f}")
            set_weights(brain, best_individual)       
    finally:
        print("Evolution completed or interrupted. Data saved.")

    return best_individual, best_individual_reward

def run_es(seed, scenario, brain, connectivity):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    es_folder = os.path.join(base_dir, f"ES_controller")
    os.makedirs(es_folder, exist_ok=True)

    scenario_folder = os.path.join(es_folder, scenario)
    os.makedirs(scenario_folder, exist_ok=True)

    seed_folder = os.path.join(scenario_folder, f"seed_{seed} - {timestamp}")
    os.makedirs(seed_folder, exist_ok=True)

    best_individual, best_individual_reward = es(seed_folder, brain, scenario, connectivity)

    if best_individual is not None:
        print("Best Individual found:")
        print(best_individual)
        print("Best Individual reward: ", best_individual_reward)

    utils.save_plot(seed_folder, scenario, seed)
    visualize_policy(best_individual, scenario, connectivity, seed_folder)


def run_seed(seed, scenario, brain, connectivity):
    utils.set_seed(seed)
    run_es(seed, scenario, brain, connectivity)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    connectivity = get_full_connectivity(robot_structure)
    processes = []
    for scenario in utils.scenarios_3_2:
        env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        sim = env.sim
        input_size = env.observation_space.shape[0]  # Observation size
        output_size = env.action_space.shape[0]  # Action size
        brain = NeuralController(input_size, output_size)

        for seed in utils.seed_list:
            p = multiprocessing.Process(target=run_seed, args=(seed, scenario, brain, connectivity))
            p.start()
            processes.append(p)


    



# ---- VISUALIZATION ----
def visualize_policy(weights, scenario, connectivity, seed_folder):
    set_weights(brain, weights)  # Load weights into the network
    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]  # Get initial state

    frames = []
    for t in range(STEPS):  
        # Update actuation before stepping
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        action = brain(state_tensor).detach().numpy().flatten() # Get action
        viewer.render('screen') 
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
            break
        frame = viewer.render('rgb_array')
        frames.append(frame)

    viewer.close()
    imageio.mimsave(seed_folder, frames, duration=0.066, optimize=True)
    env.close()