import os
import utils

def plot_all_seed_folders(base_dir="gen_data_structures", specific_folder="ES_Structure"):
    specific_folder = os.path.join(base_dir, specific_folder)
    for s in utils.scenarios:
        global scenario
        global seed 
        scenario = s
        scenario_folder = os.path.join(specific_folder, s)
        if not os.path.exists(scenario_folder):
            print(f"Scenario folder {scenario_folder} does not exist.")
            continue
        
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
# plot_all_seed_folders(input("Enter the base directory: "), input("Enter the specific folder: "))
# plot_all_seed_folders("gen_data_structures", "ES_Structure")
plot_all_seed_folders("gen_data_controllers", "CMA_ES_controller")