import random
import re
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
from evogym import EvoViewer, get_full_connectivity
import imageio
import pandas as pd
import torch
from fixed_controllers import *
from evogym.envs import *
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected

import csv
import os
from datetime import datetime
import json

from neural_controller import NeuralController, set_weights

# ---- SIMULATE BEST ROBOT ----
def simulate_best_robot(robot_structure, scenario=None, steps=500, controller = alternating_gait):
    
    connectivity = get_full_connectivity(robot_structure)
    #if not isinstance(connectivity, np.ndarray):
    #    connectivity = np.zeros(robot_structure.shape, dtype=int)
    env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
    env.reset()
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    action_size = sim.get_dim_action_space('robot')  # Get correct action size
    t_reward = 0
    
    for t in range(200):  # Simulate for 200 timesteps
        # Update actuation before stepping
        actuation = controller(action_size,t)

        ob, reward, terminated, truncated, info = env.step(actuation)
        t_reward += reward
        if terminated or truncated:
            env.reset()
            break
        viewer.render('screen')
    viewer.close()
    env.close()

    return t_reward #(max_height - initial_height) #-  abs(np.mean(positions[0, :])) # Max height gained is jump performance


def create_gif(robot_structure, filename='best_robot.gif', duration=0.066, scenario=None, steps=500, controller=alternating_gait):
    try:
        """Create a smooth GIF of the robot simulation at 30fps."""
        connectivity = get_full_connectivity(robot_structure)
        #if not isinstance(connectivity, np.ndarray):
        #    connectivity = np.zeros(robot_structure.shape, dtype=int)
        env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        t_reward = 0

        frames = []
        for t in range(200):
            actuation = controller(action_size,t)
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break
            frame = viewer.render('rgb_array')
            frames.append(frame)

        viewer.close()
        imageio.mimsave(filename, frames, duration=duration, optimize=True)
    except ValueError as e:
        print('Invalid')


def create_gifs(structure, scenario, controller, steps=500, filename="run.gif"):
    connectivity = get_full_connectivity(structure)
    env = gym.make(scenario, body=structure, connections=connectivity, max_episode_steps=steps, render_mode="rgb_array")
    
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]

    brain = NeuralController(input_size, output_size)
    set_weights(brain, controller)

    frames = []
    state = env.reset()[0]
    for _ in range(steps):
        frame = env.render()
        frames.append(frame)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    env.close()
    imageio.mimsave(filename, frames, duration=0.066, optimize=True)  # 30 FPS

    
# Added by us
scenarios_3_1 = ["Walker-v0", "BridgeWalker-v0"]
scenarios_3_2 = ["DownStepper-v0", "ObstacleTraverser-v0"]
scenarios_3_3 = ["GapJumper-v0", "CaveCrawler-v0"]
scenarios = ["DownStepper-v0", "BridgeWalker-v0", "ObstacleTraverser-v0", "GapJumper-v0", "CaveCrawler-v0"]
seed_list = [42, 123, 2025, 8675309, 123456789]

def set_seed(seed: int):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def save_structures(generation, population_with_fitness, filename="evolution_data.csv"):
    """Save generation data to a CSV file."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(["Generation", "Individual", "Fitness", "RobotStructure"])
        
        for idx, (robot, fitness) in enumerate(population_with_fitness):
            robot_str = '|'.join([','.join(map(str, row)) for row in robot])
            writer.writerow([generation, idx, fitness, robot_str])

def save_structure(population_with_fitness, filename="evolution_data.csv"):
    """Save generation data to a CSV file."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(["Individual", "Fitness", "RobotStructure"])
        
        for idx, (robot, fitness) in enumerate(population_with_fitness):
            robot_str = '|'.join([','.join(map(str, row)) for row in robot])
            writer.writerow([idx, fitness, robot_str])


def save_controllers(generation, population_with_fitness, filename="evolution_data.csv"):
    """Save generation data to a CSV file."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(["Generation", "Individual", "Fitness", "Controller"])
            
        for idx, (controller, fitness) in enumerate(population_with_fitness):
            controller_str = ";".join(
                [",".join(map(str, w.flatten())) 
                 for w in controller]
            )
            writer.writerow([generation, idx, fitness, controller_str])

def save_controller(population_with_fitness, filename="evolution_data.csv"):
    """Save generation data to a CSV file."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(["Individual", "Fitness", "Controller"])
            
        for idx, (controller, fitness) in enumerate(population_with_fitness):
            controller_str = ";".join(
                [",".join(map(str, w.flatten())) 
                 for w in controller]
            )
            writer.writerow([idx, fitness, controller_str])


def save_pairing(pairing, filename):
    """Save generation data to a CSV file."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(["Index", "Fitness", "Structure", "Controller"])
            
        for idx, (structure, controller, fitness) in enumerate(pairing):
            structure_str = '|'.join([','.join(map(str, row)) for row in structure])

            controller_str = ";".join(
                [",".join(map(str, w.flatten())) 
                 for w in controller]
            )

            writer.writerow([idx, fitness, structure_str, controller_str])





def collect_fitness_data(seed_folder):
    gen_files = [f for f in os.listdir(seed_folder) if re.match(r'gen_\d+\.csv', f)]
    fitness_summary = []

    for file in sorted(gen_files, key=lambda x: int(re.search(r'\d+', x).group())):
        gen_num = int(re.search(r'\d+', file).group())
        df = pd.read_csv(os.path.join(seed_folder, file))
        fitness_values = df['Fitness'].values

        fitness_summary.append({
            'Generation': gen_num,
            'Best': max(fitness_values),
            'Average': sum(fitness_values) / len(fitness_values),
            'Worst': min(fitness_values)
        })

    return pd.DataFrame(fitness_summary)

def save_plot(seed_folder, scenario, seed): 
    fitness_df = collect_fitness_data(seed_folder)
    if fitness_df.empty:
        print(f"No data to plot in {seed_folder}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(fitness_df['Generation'], fitness_df['Best'], label='Best Fitness', marker='o')
    plt.plot(fitness_df['Generation'], fitness_df['Average'], label='Average Fitness', marker='x')
    plt.plot(fitness_df['Generation'], fitness_df['Worst'], label='Worst Fitness', marker='s')

    plt.title("ES | " + scenario  + f" | Seed {seed} Fitness Over Generations")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.ylim(-5, 12)  # <--- This line sets the y-axis range
    plt.tight_layout()


    # Save the plot
    plot_path = os.path.join(seed_folder, "fitness_plot.png")
    if os.path.exists(plot_path):
        os.remove(plot_path)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to: {plot_path}")