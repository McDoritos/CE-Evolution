import os
import pandas as pd
import matplotlib.pyplot as plt
import re

import utils

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

def plot_fitness_over_generations(seed_folder):
    global scenario
    global seed 
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
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(seed_folder, "fitness_plot.png")
    if os.path.exists(plot_path):
        os.remove(plot_path)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to: {plot_path}")




def plot_all_seed_folders(base_dir="gen_data_structures"):
    es_folder = os.path.join(base_dir, f"ES_Structure")
    os.makedirs(es_folder, exist_ok=True)
    
    for s in utils.scenarios:
        global scenario
        global seed 
        scenario = s
        scenario_folder = os.path.join(es_folder, s)
        os.makedirs(scenario_folder, exist_ok=True)
        
        for entry in os.listdir(scenario_folder):
            seed = int(entry.split("_")[1].split(" - ")[0])
            seed_folder = os.path.join(scenario_folder, entry)  # FIXED LINE
            if os.path.isdir(seed_folder) and entry.startswith("seed_"):
                print(f"Processing: {seed_folder}")
                plot_fitness_over_generations(seed_folder)

global scenario
global seed 
scenario = ""
seed = 0
# Run the whole thing
plot_all_seed_folders()