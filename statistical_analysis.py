import os
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm.auto import tqdm  # Provides auto-detection for notebook/console

# Configuration
ALPHA = 0.20  # 80% confidence level
BASE_DIRS = ["gen_data_controllers", "gen_data_structures"]
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_process_data(base_dirs):
    """Load and preprocess data from all base directories with progress tracking"""
    all_data = []
    
    print("Loading data from directories...")
    for base_dir in tqdm(base_dirs, desc="Base directories"):
        algorithms = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        for algoritmo in tqdm(algorithms, desc=f"Algorithms in {os.path.basename(base_dir)}", leave=False):
            alg_path = os.path.join(base_dir, algoritmo)
            tasks = [t for t in os.listdir(alg_path) if os.path.isdir(os.path.join(alg_path, t))]
            
            for tarefa in tqdm(tasks, desc=f"Tasks in {algoritmo}", leave=False):
                tarefa_path = os.path.join(alg_path, tarefa)
                seeds = [s for s in os.listdir(tarefa_path) if os.path.isdir(os.path.join(tarefa_path, s))]
                
                for seed_dir in tqdm(seeds, desc=f"Seeds in {tarefa}", leave=False):
                    seed_path = os.path.join(tarefa_path, seed_dir)
                    try:
                        seed = int(seed_dir.split('_')[1].split(" -")[0])
                    except (IndexError, ValueError):
                        continue
                    
                    gen_files = list(glob(os.path.join(seed_path, "gen_*.csv")))
                    for gen_file in tqdm(gen_files, desc="Generations", leave=False):
                        try:
                            geracao = int(os.path.basename(gen_file).split("_")[1].split(".")[0])
                            df = pd.read_csv(gen_file)
                            
                            if 'Fitness' not in df.columns:
                                continue
                            
                            df["Algoritmo"] = algoritmo
                            df["Tarefa"] = tarefa
                            df["Seed"] = seed
                            df["Geração"] = geracao
                            all_data.append(df)
                        except Exception as e:
                            tqdm.write(f"Error processing {gen_file}: {str(e)}")
                            continue
    
    if not all_data:
        raise ValueError("No valid data found in the specified directories")
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df['Fitness'] = pd.to_numeric(full_df['Fitness'], errors='coerce')
    tqdm.write(f"\nLoaded data with {len(full_df)} records")
    return full_df

def analyze_task(df, task_name):
    """Perform complete analysis for one task and determine best algorithm(s)"""
    tqdm.write(f"\n{'='*50}\nAnalyzing: {task_name}\n{'='*50}")
    
    # Get best fitness per seed
    df_best = []
    for (algo, seed), group in tqdm(df.groupby(['Algoritmo', 'Seed']),
                                   desc="Processing seeds",
                                   total=df[['Algoritmo', 'Seed']].drop_duplicates().shape[0]):
        max_fitness = group['Fitness'].max()
        if pd.notna(max_fitness):  # Only include valid fitness values
            df_best.append({
                'Algoritmo': algo,
                'Seed': seed,
                'Fitness': max_fitness
            })
    
    if not df_best:
        tqdm.write(f"Warning: No valid fitness data found for {task_name}")
        return None
    
    df_best = pd.DataFrame(df_best)
    
    # Save data for later inspection
    csv_path = os.path.join(OUTPUT_DIR, f"best_fitness_{task_name}.csv")
    df_best.to_csv(csv_path, index=False)
    
    # Calculate mean fitness for each algorithm
    algo_means = df_best.groupby('Algoritmo')['Fitness'].agg(['mean', 'std', 'count'])
    best_mean = algo_means['mean'].max()
    
    # ANOVA to check for any differences
    model = ols('Fitness ~ C(Algoritmo)', data=df_best).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    p_value = anova.loc['C(Algoritmo)', 'PR(>F)']
    
    # Determine best algorithm(s)
    if p_value < ALPHA:  # Significant differences exist
        algorithms = algo_means.index.tolist()
        best_algorithms = []
        best_candidates = algo_means[algo_means['mean'] == best_mean].index.tolist()
        
        if len(best_candidates) == 1:
            best_algo = best_candidates[0]
            is_significantly_better = True
            
            for other_algo in algorithms:
                if other_algo == best_algo:
                    continue
                    
                group1 = df_best[df_best['Algoritmo'] == best_algo]['Fitness']
                group2 = df_best[df_best['Algoritmo'] == other_algo]['Fitness']
                _, p_val = ttest_ind(group1, group2, equal_var=True)
                
                if p_val >= ALPHA:
                    is_significantly_better = False
                    best_algorithms.append(other_algo)
            
            best_algorithms = [best_algo] if is_significantly_better else best_candidates + best_algorithms
        else:
            best_algorithms = best_candidates.copy()
            for i in range(len(best_candidates)):
                for j in range(i+1, len(best_candidates)):
                    group1 = df_best[df_best['Algoritmo'] == best_candidates[i]]['Fitness']
                    group2 = df_best[df_best['Algoritmo'] == best_candidates[j]]['Fitness']
                    _, p_val = ttest_ind(group1, group2, equal_var=True)
                    
                    if p_val < ALPHA and best_candidates[j] in best_algorithms:
                        best_algorithms.remove(best_candidates[j])
    else:  # No significant differences
        threshold = 0.9 * best_mean
        best_algorithms = algo_means[algo_means['mean'] >= threshold].index.tolist()
    
    # Generate report
    report = {
        'task': task_name,
        'best_algorithms': best_algorithms,
        'best_mean_fitness': best_mean,
        'all_algorithms': algo_means.to_dict('index'),
        'anova_p_value': p_value,
        'significant_difference': p_value < ALPHA
    }
    
    # Visualization highlighting best algorithm(s)
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=df_best, x='Algoritmo', y='Fitness', order=algo_means.index)
    
    # Highlight best algorithm(s) - more robust implementation
    for i, algo in enumerate(algo_means.index):
        if algo in best_algorithms:
            for j, box in enumerate(ax.artists):
                if ax.get_xticklabels()[j].get_text() == algo:
                    box.set_facecolor('lightgreen')
                    box.set_edgecolor('green')
                    box.set_alpha(0.7)
                    break
    
    plt.title(f"Best Fitness Distribution - {task_name}\nBest: {', '.join(best_algorithms)}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"best_fitness_{task_name}.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Print summary
    tqdm.write(f"\n=== Analysis Summary for {task_name} ===")
    tqdm.write(f"ANOVA p-value: {p_value:.4f} ({'Significant' if p_value < ALPHA else 'No significant'} difference)")
    tqdm.write(f"Best mean fitness: {best_mean:.4f}")
    tqdm.write("Best algorithm(s): " + ", ".join(best_algorithms))
    tqdm.write("\nAll algorithm performances:")
    tqdm.write(algo_means.sort_values('mean', ascending=False).to_string())
    
    return report

def main():
    # Load and process all data
    full_df = load_and_process_data(BASE_DIRS)
    
    # Analyze each task and collect reports
    reports = []
    tasks = full_df['Tarefa'].unique()
    
    for task in tqdm(tasks, desc="Analyzing tasks"):
        task_df = full_df[full_df['Tarefa'] == task]
        report = analyze_task(task_df, task)
        reports.append(report)
    
    # Also analyze by algorithm type (controllers vs structures)
    for base_dir in BASE_DIRS:
        dir_name = os.path.basename(base_dir)
        dir_df = full_df[full_df['Tarefa'].isin(os.listdir(base_dir))]
        if not dir_df.empty:
            report = analyze_task(dir_df, f"All_{dir_name}")
            reports.append(report)
    
    # Generate final summary report
    final_report = pd.DataFrame([{
        'Task': r['task'],
        'Best Algorithm(s)': ", ".join(r['best_algorithms']),
        'Best Mean Fitness': r['best_mean_fitness'],
        'Significant Difference': "Yes" if r['significant_difference'] else "No",
        'ANOVA p-value': r['anova_p_value']
    } for r in reports])
    
    # Save final report
    report_path = os.path.join(OUTPUT_DIR, "final_algorithm_ranking.csv")
    final_report.sort_values(by=['Task']).to_csv(report_path, index=False)
    tqdm.write(f"\nSaved final report to {report_path}")
    
    # Print final recommendations
    tqdm.write("\n=== FINAL RECOMMENDATIONS ===")
    for task in final_report['Task'].unique():
        task_data = final_report[final_report['Task'] == task].iloc[0]
        tqdm.write(f"\nFor {task}:")
        tqdm.write(f"  • Best algorithm(s): {task_data['Best Algorithm(s)']}")
        tqdm.write(f"  • Mean fitness: {task_data['Best Mean Fitness']:.4f}")
        if task_data['Significant Difference'] == "Yes":
            tqdm.write("  • Clear winner (statistically significant)")
        else:
            tqdm.write("  • No significant difference between top performers")

if __name__ == "__main__":
    main()
    print("\nAnalysis complete! Best algorithms identified for each scenario.")