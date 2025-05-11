import os
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load and preprocess data (same as before)
base_dirs = ["gen_data_controllers", "gen_data_structures"]

def load_data(base_dir):
    all_data = []
    for algoritmo in os.listdir(base_dir):
        alg_path = os.path.join(base_dir, algoritmo)
        if not os.path.isdir(alg_path):
            continue
        
        for tarefa in os.listdir(alg_path):
            tarefa_path = os.path.join(alg_path, tarefa)
            
            for seed_dir in os.listdir(tarefa_path):
                seed_path = os.path.join(tarefa_path, seed_dir)
                seed = int(seed_dir.split('_')[1].split(" -")[0])
                
                for gen_file in glob(os.path.join(seed_path, "gen_*.csv")):
                    geracao = int(os.path.basename(gen_file).split("_")[1].split(".")[0])
                    df = pd.read_csv(gen_file)
                    
                    if 'Fitness' not in df.columns:
                        raise ValueError(f"Arquivo {gen_file} não contém coluna 'Fitness'")
                    
                    df["Algoritmo"] = algoritmo
                    df["Tarefa"] = tarefa
                    df["Seed"] = seed
                    df["Geração"] = geracao
                    all_data.append(df)
    return all_data
for base_dir in base_dirs:
    all_data = load_data(base_dir)   
    full_df = pd.concat(all_data, ignore_index=True)
    full_df['Fitness'] = pd.to_numeric(full_df['Fitness'], errors='coerce') 

#     # Filter for the two algorithms
#     df_filtered = full_df[full_df['Algoritmo'].isin(['GA_Structure 100x30', 'ES_Structure'])]

#     # Get the BEST fitness per seed (across all generations)
#     df_best_per_seed = df_filtered.groupby(['Algoritmo', 'Seed'])['Fitness'].max().reset_index()

#     # Visualize the best fitness distribution
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(data=df_best_per_seed, x='Algoritmo', y='Fitness')
#     plt.title("Best Fitness per Seed (Max Across Generations)")
#     plt.savefig("best_fitness_per_seed.png")
#     plt.show()

#     # ANOVA: Compare algorithms based on best fitness
#     model = ols('Fitness ~ C(Algoritmo)', data=df_best_per_seed).fit()
#     anova = sm.stats.anova_lm(model, typ=2)
#     print("\nANOVA (Best Fitness per Seed):")
#     print(anova)

#     # Independent t-test (same as before)
#     ga = df_best_per_seed[df_best_per_seed['Algoritmo'] == 'GA_Structure 100x30']['Fitness']
#     es = df_best_per_seed[df_best_per_seed['Algoritmo'] == 'ES_Structure']['Fitness']
#     t_stat, p_value = ttest_ind(ga, es, equal_var=True)
#     print(f"\nT-test (Best Fitness per Seed):\nT-statistic: {t_stat:.4f}\nP-value: {p_value:.4e}")

#     # Post-hoc Tukey HSD if ANOVA is significant (p < 0.05)
#     if anova.loc['C(Algoritmo)', 'PR(>F)'] < 0.05:
#         tukey = pairwise_tukeyhsd(df_best_per_seed['Fitness'], df_best_per_seed['Algoritmo'])
#         print("\nTukey HSD Post-hoc Test:")
#         print(tukey)
#     else:
#         print("\nNo significant difference (ANOVA p > 0.05). Skipping post-hoc.")

#     # Save results
#     df_best_per_seed.to_csv("best_fitness_per_seed.csv", index=False)