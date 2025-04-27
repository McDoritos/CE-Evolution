import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *

from tqdm import trange
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

def mutate(individual, noise_std=0.1):
    """Mutate the weights by adding Gaussian noise."""
    mutated = []
    for param in individual:
        noise = np.random.normal(0, noise_std, param.shape)
        mutated.append(param + noise)
    return mutated

# ---- EVOLUTIONARY STRATEGY ALGORITHM ----
population = [ [np.random.randn(*param.shape) for param in brain.parameters()] for _ in range(POPULATION_SIZE) ]
best_individual = None
best_individual_reward = -np.inf

best_rewards = []
mean_rewards = []

for gen in trange(NUM_GENERATIONS, desc="Evolving ES", unit="gen"):

    offspring = [ mutate(random.choice(population)) for _ in range(OFFSPRING_SIZE) ]
    combined_population = population + offspring

    # SEM multiprocessing, avaliação em loop normal
    fitness_values = [evaluate_fitness(individual) for individual in combined_population]

    population_with_fitness = list(zip(combined_population, fitness_values))
    population_with_fitness = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)

    if population_with_fitness[0][1] > best_individual_reward:
        best_individual_reward = population_with_fitness[0][1]
        best_individual = population_with_fitness[0][0].copy()

    avg_fitness = sum(fitness for _, fitness in population_with_fitness) / len(population_with_fitness)
    best_rewards.append(best_individual_reward)
    mean_rewards.append(avg_fitness)

    population = [ind for ind, _ in population_with_fitness[:POPULATION_SIZE]]

    print(f"Generation {gen + 1}: Best Reward = {best_individual_reward:.4f}, Mean Reward = {avg_fitness:.4f}")

print("Best Fitness Achieved: ", best_individual_reward)
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