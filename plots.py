import os
import utils





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
                utils.save_plot(seed_folder, scenario, seed)

global scenario
global seed 
scenario = ""
seed = 0
# Run the whole thing
plot_all_seed_folders()