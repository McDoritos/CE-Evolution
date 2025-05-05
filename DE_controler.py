import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *

from tqdm import trange
import utils
from datetime import datetime

os.makedirs("gen_data_controllers", exist_ok=True)

NUM_GENERATIONS = 100  # Number of generations to evolve
STEPS = 500
SCENARIO = 'DownStepper-v0'
SEED = 42
POPULATION_SIZE = 10 # 10 > 20 
CROSSOVER_PER = 0.7
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


# ---- DIFERENTIAL EVOLUTION ALGORITHM ----
try:
    # 1 - Generate a random population
    population = []
    population_fitness = []
    best_rewards = []
    mean_rewards = []
    best_individual = None
    best_individual_reward = -np.inf

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join("gen_data_controllers", f"DE-Controller_evolution-data_{timestamp}.csv")

    for i in range(POPULATION_SIZE):

        individual = [np.random.randn(*param.shape) for param in brain.parameters()]
        population.append(individual)
        reward = evaluate_fitness(individual)
        population_fitness.append(reward)

        if reward > best_individual_reward:
            best_individual = individual.copy()
            best_individual_reward = reward

    for generation in trange(NUM_GENERATIONS, desc="Evolving DE", unit="gen"):
        # 2 - Mutate child with different weighted solutions
        mutated_pop = []
        best_idx = np.argmax(population_fitness)
        for i in range(len(population)):
            random_idx = random.sample(range(len(population)),2)
            
            lam = 0.5 * (1 + random.random())
            variant = [np.array(population[best_idx][k]) + lam * (np.array(population[random_idx[0]][k]) - np.array(population[random_idx[1]][k])) for k in range(len(population[random_idx[0]]))]
            mutated_pop.append(variant)

        # 3 - Mix mutant vectors with target vectors to create trial solutions
        trial_pop = []
        for i, j in zip(population, mutated_pop):
            i_rand = random.randint(0, len(population)-1)
            trial = np.zeros_like(i)
            for k in range(len(i)):
                if random.random() < CROSSOVER_PER or k == i_rand:
                    trial[k] = j[k]
                else:
                    trial[k] = i[k]
            trial_pop.append(trial)

        # 4 - Choose the better solution between the target and trial vectors
        new_population = []
        new_population_rewards = []
        for i, j in zip(population, trial_pop):
            pop_reward = evaluate_fitness(i)
            trial_reward = evaluate_fitness(j)

            if trial_reward > pop_reward:
                new_population.append(j)
                new_population_rewards.append(trial_reward)
                if trial_reward > best_individual_reward:
                    best_individual = j.copy()
                    best_individual_reward = trial_reward
            else:
                new_population.append(i)
                new_population_rewards.append(pop_reward)
                if pop_reward > best_individual_reward:
                    best_individual = i.copy()
                    best_individual_reward = pop_reward

        best_reward = max(new_population_rewards)
        population = new_population.copy()
        population_fitness = new_population_rewards.copy()
        
        population_with_fitness = list(zip(population, population_fitness))
        population_with_fitness = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)

        # Salva a geração atual
        utils.save_controllers(generation+1, population_with_fitness, csv_filename)

        best_rewards.append(best_reward)
        mean_rewards.append(sum(new_population_rewards) / len(new_population_rewards))

        print(f"Generation {generation + 1}: Best Reward = {best_reward} Mean Reward = {sum(new_population_rewards) / len(new_population_rewards)}")

    print("Best Fitness: ", best_individual_reward)
    set_weights(brain, best_individual) 
except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Saving current state...")
    population_with_fitness = list(zip(population, population_fitness))
    population_with_fitness = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
    utils.save_controllers(generation+1, population_with_fitness, csv_filename)
    
    if best_individual is not None:
        print(f"Best reward achieved: {best_individual_reward:.4f}")
        set_weights(brain, best_individual)       
finally:
    print("Evolution completed or interrupted. Data saved.")


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
while i == 0 :
    visualize_policy(best_individual)