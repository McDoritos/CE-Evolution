import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *

from tqdm import trange
import cma
#import multiprocessing


NUM_GENERATIONS = 100  # Number of generations to evolve
STEPS = 500
SCENARIO = 'DownStepper-v0'
SEED = 42
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

connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]  # Observation size
output_size = env.action_space.shape[0]  # Action size

brain = NeuralController(input_size, output_size)

# ---- FITNESS FUNCTION ----
def evaluate_fitness(weights, view=False):
        set_weights(brain, weights)  # Load weights into the network
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
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

weight_shapes = [p.shape for p in brain.parameters()]

def cma_fitness_wrapper(vector):
    """adapts the fitness fuction to the CMA-ES"""
    weights = vector_to_weights(vector, weight_shapes)
    return -evaluate_fitness(weights)  # CMA-ES minimiza, então é invertida a fitness

# Initializing CMA-ES
initial_params = weights_to_vector([np.random.randn(*s) for s in weight_shapes])
es = cma.CMAEvolutionStrategy(initial_params, 0.5, {'seed': SEED})

best_rewards = []
mean_rewards = []
best_individual = None
best_individual_reward = -np.inf

for gen in trange(NUM_GENERATIONS, desc="CMA-ES Optimization"):
    # 1 - Generates solutions
    solutions = es.ask()

    # 2 - Evaluates solutions
    fitness = [cma_fitness_wrapper(x) for x in solutions]

    # 3 - Updates the strategy with the results (covariance matrix and sigma (step length))
    es.tell(solutions, fitness)

    # 4 - Anotates the best fitness, the mean and the best individual
    current_best = -es.best.f
    current_mean = -np.mean(fitness)
    
    if current_best > best_individual_reward:
        best_individual_reward = current_best
        best_individual = vector_to_weights(es.best.x, weight_shapes)
    
    best_rewards.append(current_best)
    mean_rewards.append(current_mean)
    
    print(f"Gen {gen+1}: Best {current_best:.2f} | Mean {current_mean:.2f} | σ={es.sigma:.3f}\n")

set_weights(brain, best_individual)

# ---- VISUALIZATION ----
def visualize_policy(weights):
    set_weights(brain, weights)  # Load weights into the network
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
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

i = 0
while i == 0:
    visualize_policy(best_individual)