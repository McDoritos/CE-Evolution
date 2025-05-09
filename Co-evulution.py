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

# Configurações
NUM_GENERATIONS = 100
STEPS = 500
SCENARIO = 'CaveCrawler-v0'  # CaveCrawler-v0
SEED = 42
POPULATION_SIZE = 20
MUTATION_RATE = 0.1
ELITE_SIZE = 5
MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)
VOXEL_TYPES = [0, 1, 2, 3, 4]  # 0: vazio, 1: rígido, 2: macio, 3: atuador+, 4: atuador-


np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

class CoEvolution:
    def __init__(self, pop_size_struc=3, offspring_size=3 , pop_size_con=20, elite=5, mutation=0.1):
        self.pop_size_struc = pop_size_struc
        self.pop_size_con = pop_size_con
        self.mutation = mutation
        self.offspring_size = offspring_size
        self.elite = 5
        # Inicializa populações
        self.pop_struc = [self.create_random_robot() for _ in range(pop_size_struc)]
        
        self.fitness_struc = []
        self.fitness_con = []
        self.pairings = []
        
        self.best_struc = None
        self.best_con = None
        self.best_fitness = -np.inf

        self.num_repeats_con = 10

    def evaluate(self, robot_structure, controller_weights, view=False):    
        #try:
            connectivity = get_full_connectivity(robot_structure)
            env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)

            input_size = env.observation_space.shape[0]
            output_size = env.observation_space.shape[0]

            brain = NeuralController(input_size, output_size)
            set_weights(brain, controller_weights)
            
            state = env.reset()[0]
            viewer = EvoViewer(env.sim) if view else None
            total_reward = 0
            
            for t in range(STEPS):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = brain(state_tensor).detach().numpy().flatten()
                
                if view:
                    viewer.render('screen')
                
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            if view:
                viewer.close()
            env.close()
            return total_reward
        #except Exception as e:
        #    print(f"Error in evaluation: {str(e)}")
        #    return -1000

    def evaluate_parallel(self, struc, weights_list):
        """Versão com multiprocessing"""
        args = [(struc, weights) for weights in weights_list]
        
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(self._evaluate_single, args)
        return results

    def _evaluate_single(self, struc, weights):
        connectivity = get_full_connectivity(struc)
        env = gym.make(SCENARIO, max_episode_steps=STEPS, 
                      body=struc, connections=connectivity)
        
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
    
    def mutate_weights(self, weights, noise_std=0.1):
        """Aplica mutação gaussiana a cada peso"""
        return [w + np.random.normal(0, noise_std, w.shape) for w in weights]

    def evolutionary_algorithm(self):
        self.fitness_struc = []
        self.fitness_con = []
        new_pairings = []
        
        offsprings_struc = [self.mutate(random.choice(self.pop_struc)) for _ in range(self.offspring_size)]
        combined_population_struc = self.pop_struc + offsprings_struc

        for struc in combined_population_struc:
            connectivity = get_full_connectivity(struc)
            env = gym.make(SCENARIO, max_episode_steps=STEPS, body=struc, connections=connectivity)
            input_size = env.observation_space.shape[0]
            output_size = env.action_space.shape[0]
            env.close()

            existing_pair = next((pair for pair in self.pairings 
                                if np.array_equal(pair[0], struc)), None)

            if existing_pair:
                # ESTRUTURA EXISTENTE - avaliação paralela
                _, best_con, previous_fit = existing_pair
                mutated_pop = [self.mutate_weights(best_con['weights']) for _ in range(10)]
                fits = self.evaluate_parallel(struc, mutated_pop)
                
                best_idx = np.argmax(fits)
                best_fit = max(fits[best_idx], previous_fit)
                best_weights = mutated_pop[best_idx] if fits[best_idx] > previous_fit else best_con['weights']
            else:
                # NOVA ESTRUTURA - avaliação paralela
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

        # Restante do método mantido igual
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

        return (self.best_struc.copy(), self.best_con.copy()), self.best_fitness


    def evolve(self, generations):
        for gen in trange(generations, desc="Co-evolving"):
            (best_struc, best_con), best_fit = self.evolutionary_algorithm()
            print(f"Generation {gen+1}: Best Fitness (global) = {self.best_fitness:.4f} ; Best Fitness (gen) = {best_fit:.4f}")
        
        return self.best_struc, self.best_con, self.best_fitness

if __name__ == "__main__":
    co_evolver = CoEvolution()
    best_struct, best_ctrl, best_fit = co_evolver.evolve(NUM_GENERATIONS)
    
    # Visualização do melhor resultado
    print("\n=== Best Solution ===")
    print(f"Best Fitness: {best_fit:.4f}")
    print("Best Structure:")
    print(best_struct)
    print("Best Controller:")
    print(best_ctrl)
    
    # Configurar rede neural para visualização
    env = gym.make(SCENARIO, body=best_struct)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    brain = NeuralController(input_size, output_size)

    # Certificar-se que best_ctrl tem os shapes corretos
    if len(best_ctrl) != len(list(brain.parameters())):
        print("Erro: Dimensões do controlador evoluído não correspondem à rede!")
        exit()

    set_weights(brain, best_ctrl)
    env.close()

    # Visualização
    env = gym.make(SCENARIO, body=best_struct, render_mode='human')
    state = env.reset()[0]
    viewer = EvoViewer(env.sim)
    viewer.track_objects('robot')

    for _ in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        try:
            action = brain(state_tensor).detach().numpy().flatten()
            viewer.render('screen')
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        except Exception as e:
            print(f"Erro durante simulação: {e}")
            break

    viewer.close()
    env.close()