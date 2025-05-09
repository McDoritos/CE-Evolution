import multiprocessing
import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *

from tqdm import trange
import cma
from datetime import datetime
import utils
#import multiprocessing

base_dir = "gen_data_controllers"
os.makedirs(base_dir, exist_ok=True)

NUM_GENERATIONS = 100  # Number of generations to evolve
STEPS = 500
SEED = 314159 #8675309 #2025 #123 #42
POPULATION_SIZE = 10 # 10 > 20 
OFFSPRING_SIZE = 20
np.random.seed(SEED)
random.seed(SEED)


robot_structure = np.array([ 
[1,3,1,0,0],
[4,1,3,2,2],
[3,4,4,4,4],
[3,0,0,3,2],
[0,0,0,0,2]
])



# ---- FITNESS FUNCTION ----
def evaluate_fitness(weights, brain, scenario, connectivity, view=False):
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

def weights_to_vector(weights):
    """convert list of weights to numpy"""
    return np.concatenate([w.flatten() for w in weights])

def vector_to_weights(vector, weight_shapes):
    """convert numpy into list of weights"""
    weights = []
    idx = 0
    for shape in weight_shapes:
        size = np.prod(shape)
        weights.append(vector[idx:idx+size].reshape(shape))
        idx += size
    return weights

def cma_fitness_wrapper(vector, weight_shapes, brain, scenario, connectivity):
    """adapts the fitness fuction to the CMA-ES"""
    weights = vector_to_weights(vector, weight_shapes)
    return -evaluate_fitness(weights, brain, scenario, connectivity)  # CMA-ES minimiza, então é invertida a fitness

def cma_es(seed_folder, weight_shapes, brain, scenario, connectivity):
    try:
        # Initializing CMA-ES
        initial_params = weights_to_vector([np.random.randn(*s) for s in weight_shapes])
        es = cma.CMAEvolutionStrategy(initial_params, 0.3, {'seed': SEED})

        best_rewards = []
        mean_rewards = []
        best_individual = None
        best_individual_reward = -np.inf

       

        for gen in trange(NUM_GENERATIONS, desc="CMA-ES Optimization"):
            # 1 - Generates solutions
            solutions = es.ask()

            # 2 - Evaluates solutions
            raw_fitness = [cma_fitness_wrapper(x, weight_shapes, brain, scenario, connectivity) for x in solutions]  # Valores negativos (CMA-ES minimiza)
            real_fitness = [-f for f in raw_fitness]  # Converte para valores reais
            
            # Prepara dados para salvar (com fitness reais)
            population_with_fitness = [
                (vector_to_weights(x, weight_shapes), f) 
                for x, f in zip(solutions, real_fitness)  # Usa real_fitness aqui
            ]
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)

            csv_filename = os.path.join(seed_folder, f"gen_{gen}.csv")
            utils.save_controller(population_with_fitness, csv_filename)

            # 3 - Updates the strategy with the results (covariance matrix and sigma (step length))
            es.tell(solutions, raw_fitness)

            # 4 - Anotates the best fitness, the mean and the best individual
            current_best = -es.best.f
            current_mean = -np.mean(raw_fitness)
            
            if current_best > best_individual_reward:
                best_individual_reward = current_best
                best_individual = vector_to_weights(es.best.x, weight_shapes)
            
            best_rewards.append(current_best)
            mean_rewards.append(current_mean)
            
            print(f"Gen {gen+1}: Best {current_best:.2f} | Mean {current_mean:.2f} | σ={es.sigma:.3f}\n")

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

# ---- VISUALIZATION ----
def visualize_policy(weights):
    set_weights(brain, weights)  # Load weights into the network
    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]  # Get initial state
    for t in range(STEPS):  
        # Update actuation before stepping
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        action = brain(state_tensor).detach().numpy().flatten() # Get action
        viewer.render('screen') 
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
            break

    viewer.close()
    env.close()

def run_CMA_ES(seed, scenario, weight_shapes, brain, connectivity):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    cma_es_folder = os.path.join(base_dir, f"CMA_ES_controller")
    os.makedirs(cma_es_folder, exist_ok=True)

    scenario_folder = os.path.join(cma_es_folder, scenario)
    os.makedirs(scenario_folder, exist_ok=True)

    seed_folder = os.path.join(scenario_folder, f"seed_{seed} - {timestamp}")
    os.makedirs(seed_folder, exist_ok=True)
    
    best_individual, best_individual_reward = cma_es(seed_folder, weight_shapes, brain, scenario, connectivity)
    if best_individual is not None:
        print("Best Individual found:")
        print(best_individual)
        print("Best Individual reward: ", best_individual_reward)

    visualize_policy(best_individual)
    utils.save_plot(seed_folder, scenario, seed)

def run_seed(seed, scenario, weight_shapes, brain, connectivity):
    utils.set_seed(seed)
    run_CMA_ES(seed, scenario, weight_shapes, brain, connectivity)

if __name__ == "__main__":
    # multiprocessing.freeze_support()
    # connectivity = get_full_connectivity(robot_structure)
    # processes = []
    # for scenario in utils.scenarios_3_2:
    #     env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    #     sim = env.sim
    #     input_size = env.observation_space.shape[0]  # Observation size
    #     output_size = env.action_space.shape[0]  # Action size
    #     brain = NeuralController(input_size, output_size)
    #     weight_shapes = [p.shape for p in brain.parameters()]
    #     for seed in utils.seed_list:
    #         p = multiprocessing.Process(target=run_seed, args=(seed, scenario, weight_shapes, brain, connectivity))
    #         p.start()
    #         processes.append(p)
    # for p in processes:
    #     p.join()
    #     processes.remove(p)
    connectivity = get_full_connectivity(robot_structure)
    seed = 123
    scenario = "DownStepper-v0"
    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    input_size = env.observation_space.shape[0]  # Observation size
    output_size = env.action_space.shape[0]  # Action size
    brain = NeuralController(input_size, output_size)
    weight_shapes = [p.shape for p in brain.parameters()]
    run_seed(seed,scenario,weight_shapes,brain,connectivity)


    