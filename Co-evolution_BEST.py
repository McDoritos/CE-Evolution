from datetime import datetime
import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
from neural_controller import *
from tqdm import trange
from fixed_controllers import *
import copy
from multiprocessing import Pool, cpu_count
import multiprocessing
import utils

# Configurações
NUM_GENERATIONS = 100
STEPS = 500
SCENARIO = 'GapJumper-v0'  # CaveCrawler-v0
MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)
VOXEL_TYPES = [0, 1, 2, 3, 4]  # 0: vazio, 1: rígido, 2: macio, 3: atuador+, 4: atuador-

class CoEvolution:
    def __init__(self, scenario, seed, seed_folder, pop_size_struc= 10, offspring_size=10 , pop_size_con=20, elite=5, mutation=0.2):
        utils.set_seed(seed)
        self.pop_size_struc = pop_size_struc
        self.pop_size_con = pop_size_con
        self.mutation = mutation
        self.offspring_size = offspring_size
        self.elite = elite
        self.scenario = scenario
        self.seed_folder = seed_folder
        # Inicializa populações
        self.pop_struc = [self.create_random_robot() for _ in range(pop_size_struc)]
        
        self.fitness_struc = []
        self.fitness_con = []
        self.pairings = []
        
        self.best_struc = None
        self.best_con = None
        self.best_fitness = -np.inf

        self.min_diversity = 0.1  # Limite mínimo de diversidade
        self.fitness_history = []  # Histórico para mutação adaptativa

    def evaluate_parallel(self, struc, weights_list):
        """Versão serial da avaliação"""
        results = []
        for weights in weights_list:
            results.append(self._evaluate_single(struc, weights))
        return results

    def _evaluate_single(self, struc, weights):
        connectivity = get_full_connectivity(struc)
        env = gym.make(self.scenario, max_episode_steps=STEPS, 
                      body=struc, connections=connectivity)
        env.metadata['render_fps'] = 30 

        brain = NeuralController(env.observation_space.shape[0], 
                               env.action_space.shape[0])
        set_weights(brain, weights)
        
        state = env.reset()[0]
        total_reward = 0
        for _ in range(STEPS):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = brain(state_tensor).detach().numpy().flatten()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
                
        env.close()
        return total_reward
    
    def create_random_robot(self):
        """Generate a valid random robot structure."""
        grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), 
                     random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
        random_robot, _ = sample_robot(grid_size)
        return random_robot
    
    def mutate(self,parent, max_attempts=5):
        shape = parent.shape
        total_cells = parent.size
        num_mutations = int(total_cells * self.mutation)

        for _ in range(max_attempts):
            child = copy.deepcopy(parent)
            child = child.flatten()
            indices = random.sample(range(total_cells), num_mutations)

            for idx in indices:
                current_value = child[idx]
                choices = [v for v in VOXEL_TYPES if v != current_value]
                child[idx] = random.choice(choices)

            child = child.reshape(shape)

            if is_connected(child) and has_actuator(child):
                return child

        #print(f"[WARNING] Mutation failed after {max_attempts} attempts. Returning original parent.")
        return parent
    
    def calculate_diversity(self):
        if len(self.pop_struc) <= 1:
            return 1.0
            
        distances = []
        for i in range(len(self.pop_struc)):
            for j in range(i+1, len(self.pop_struc)):
                diff = np.mean(self.pop_struc[i] != self.pop_struc[j])
                distances.append(diff)
        
        return np.mean(distances)

    def mutate_weights(self, weights, base_noise_std=0.1):
        noise_std = base_noise_std
    
        if len(self.fitness_history) > 10:
            recent_max = max(self.fitness_history[-10:])
            recent_min = min(self.fitness_history[-10:])
            recent_avg = np.mean(self.fitness_history[-10:])
            recent_improvement = recent_max - recent_min
            
            stagnation_threshold = 0.01 * recent_avg
            
            if recent_improvement < stagnation_threshold:
                # Aumenta mutação se estagnado
                noise_std = min(base_noise_std * 2, 0.4)
                print(f"Stagnation detected - increasing mutation to {noise_std:.3f}")
        
        mutated = []
        for w in weights:
            mutation = np.random.normal(0, noise_std, w.shape)
            new_w = w + mutation
            mutated.append(new_w)
        
        return mutated

    def evolutionary_algorithm(self):
        self.fitness_struc = []
        self.fitness_con = []
        new_pairings = []
        
        offsprings_struc = [self.mutate(random.choice(self.pop_struc)) for _ in range(self.offspring_size)]
        combined_population_struc = self.pop_struc + offsprings_struc

        for struc in combined_population_struc:
            connectivity = get_full_connectivity(struc)
            env = gym.make(self.scenario, max_episode_steps=STEPS, body=struc, connections=connectivity)
            input_size = env.observation_space.shape[0]
            output_size = env.action_space.shape[0]
            env.close()

            existing_pair = next((pair for pair in self.pairings if np.array_equal(pair[0], struc)), None)

            if existing_pair:
                _, best_con, previous_fit = existing_pair
                mutated_pop = [self.mutate_weights(best_con['weights']) for _ in range(10)]
                fits = self.evaluate_parallel(struc, mutated_pop)
                
                best_idx = np.argmax(fits)
                best_fit = max(fits[best_idx], previous_fit)
                best_weights = mutated_pop[best_idx] if fits[best_idx] > previous_fit else best_con['weights']
            else:
                brain = NeuralController(input_size, output_size)
                initial_weights = [np.random.randn(*p.shape) for p in brain.parameters()]
                mutated_pop = [self.mutate_weights(initial_weights) for _ in range(10)]
                fits = self.evaluate_parallel(struc, mutated_pop)
                
                best_idx = np.argmax(fits)
                best_fit = fits[best_idx]
                best_weights = mutated_pop[best_idx]

            best_controller = {
                'weights': best_weights,
                'input_size': input_size,
                'output_size': output_size,
                'fitness': best_fit
            }

            new_pairings.append((struc, best_controller, best_fit))
            self.fitness_struc.append(best_fit)
            self.fitness_con.append(best_fit)

        self.pairings = new_pairings.copy()    
        
        sorted_population = sorted(zip(self.fitness_struc, combined_population_struc), 
                            key=lambda x: x[0], reverse=True)
        self.pop_struc = [x for _, x in sorted_population[:self.pop_size_struc]]

        if self.pairings:
            current_best = max(self.pairings, key=lambda x: x[2])
            if current_best[2] > self.best_fitness:
                self.best_struc, self.best_con, self.best_fitness = current_best
                self.best_struc = self.best_struc.copy()
                self.best_con = self.best_con.copy()
                self.fitness_history.append(self.best_fitness)

        return (self.best_struc.copy(), self.best_con.copy()), self.best_fitness

    def evolve(self, generations):
        for gen in trange(generations, desc="Co-evolving"):
            (best_struc, best_con), best_fit = self.evolutionary_algorithm()

            gif_folder = os.path.join(self.seed_folder, f"gif.gif")
            utils.create_gif_coev(best_struc, scenario=self.scenario, controller=best_con, steps=STEPS, filename=gif_folder)
            csv_filename = os.path.join(self.seed_folder, f"gen_{gen}.csv")
            utils.save_pairing(self.pairings, csv_filename)

            print(f"Generation {gen+1}: Best Fitness (global) = {self.best_fitness:.4f}")
        
        return self.best_struc, self.best_con, self.best_fitness
    
def run_seed(seed, scenario):
    utils.set_seed(seed)
    base_dir = "gen_data_coevolution"
    os.makedirs("gen_data_coevolution", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    specific_folder = os.path.join(base_dir, f"CoEvolution")
    os.makedirs(specific_folder, exist_ok=True)

    scenario_folder = os.path.join(specific_folder, scenario)
    os.makedirs(scenario_folder, exist_ok=True)

    seed_folder = os.path.join(scenario_folder, f"seed_{seed} - {timestamp}")
    os.makedirs(seed_folder, exist_ok=True)

    co_evolver = CoEvolution(scenario, seed, seed_folder)
    best_structure, best_controller, best_fitness = co_evolver.evolve(NUM_GENERATIONS)

    # Visualização do melhor resultado
    print("\n=== Best Solution ===")
    print(f"Best Fitness: {best_fitness:.4f}")
    print("Best Structure:")
    print(best_structure)
    print("Best Controller:")
    print(best_controller)

    utils.create_gif_coev(best_structure, scenario=scenario, controller=best_controller, steps=STEPS)   

if __name__ == "__main__":
    multiprocessing.freeze_support()
    processes = []
    for scenario in utils.scenarios_3_3:
        for seed in utils.seed_list:
            p = multiprocessing.Process(target=run_seed, args=(seed, scenario))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()