import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
from neural_controller import *
from tqdm import trange
from fixed_controllers import *

# Configurações
NUM_GENERATIONS = 100
STEPS = 500
SCENARIO = 'GapJumper-v0'  # CaveCrawler-v0
SEED = 42
POPULATION_SIZE = 20
MUTATION_RATE = 0.05
ELITE_SIZE = 5
MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)
VOXEL_TYPES = [0, 1, 2, 3, 4]  # 0: vazio, 1: rígido, 2: macio, 3: atuador+, 4: atuador-


np.random.seed(SEED)
random.seed(SEED)

class CoEvolution:
    def __init__(self, pop_size_struc=20, pop_size_con=20, elite=5, mutation=0.05, offspring_size_con=20):
        self.pop_size_struc = pop_size_struc
        self.pop_size_con = pop_size_con
        self.offspring_size_con = offspring_size_con
        self.elite = elite
        self.mutation = mutation
        
        # Inicializa populações
        self.pop_struc = [self.create_random_robot() for _ in range(pop_size_struc)]
        env = gym.make(SCENARIO, body=self.pop_struc[0])  # Usa a primeira estrutura como exemplo
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        self.brain_template = NeuralController(input_size, output_size)
        self.pop_con = [[np.random.randn(*p.shape) for p in self.brain_template.parameters()] 
                    for _ in range(pop_size_con)]
        env.close()
        
        self.fitness_struc = []
        self.fitness_con = []
        self.pairings = []
        
        self.best_struc = None
        self.best_con = None
        self.best_fitness = -np.infz

    def evaluate_struc(self, robot_structure, controller_weights, view=False):    
        try:
            connectivity = get_full_connectivity(robot_structure)
            env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
            input_size = env.observation_space.shape[0]
            output_size = env.action_space.shape[0]
            
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
        except Exception as e:
            return 0.0
        
    def evaluate_con(self, controller_weights, robot_structure, view=False):
        return self.evaluate_struc(robot_structure, controller_weights, view)
    
    def create_random_robot(self):
        """Generate a valid random robot structure."""
        grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), 
                     random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
        random_robot, _ = sample_robot(grid_size)
        return random_robot
    
    def has_actuator(self, structure):
        return np.any((structure == 3) | (structure == 4))
    
    def crossover_struc(self, parent1, parent2, max_attempts=10):
        for _ in range(max_attempts):
            mask = np.random.randint(0, 2, size=parent1.shape)  
            child = np.where(mask, parent2, parent1)  
            
            if is_connected(child) and self.has_actuator(child):
                return child
        
        return parent1 if random.random() < 0.5 else parent2

    def tournament_selection(self, population, fitness_values, k=5):
        selected_indices = random.sample(range(len(population)), k)
        selected = [(population[i], fitness_values[i]) for i in selected_indices]
        selected.sort(key=lambda x: x[1], reverse=True)
        return selected[0][0], selected[1][0]

    def mutate_struc(self, individual, max_attempts=5):
        original = individual.copy()
        
        for _ in range(max_attempts):
            mutated = individual.copy()
            for i in range(mutated.shape[0]):
                for j in range(mutated.shape[1]):
                    if random.random() < self.mutation:
                        mutated[i,j] = random.choice(VOXEL_TYPES)
            
            if is_connected(mutated) and self.has_actuator(mutated):
                return mutated
        
        return original
    
    def mutate_con(self, individual, noise_std=0.1):
        return [param + np.random.normal(0, noise_std, param.shape) for param in individual]
    
    def bestPairing(self):
        self.fitness_struc = []
        self.fitness_con = []
        self.pairings = []
        
        # 1 - Evaluate structures with best controller
        for structure in self.pop_struc:
            if len(self.fitness_con) == 0:
                partner = random.choice(self.pop_con)
            else:
                best_con_idx = np.argmax(self.fitness_con)
                partner = self.pop_con[best_con_idx]
            
            fit = self.evaluate_struc(structure, partner)
            self.fitness_struc.append(fit)
            self.pairings.append((structure, partner, fit))

        # 2 - Evaluate controllers with best structure
        for controller in self.pop_con:
            if len(self.fitness_struc) == 0:
                partner = random.choice(self.pop_struc)
            else:
                best_struc_idx = np.argmax(self.fitness_struc)
                partner = self.pop_struc[best_struc_idx]
            
            fit = self.evaluate_con(controller, partner)
            self.fitness_con.append(fit)
            self.pairings.append((partner, controller, fit))

        # 3 - Find best pair
        best_struc, best_con, best_fit = max(self.pairings, key=lambda x: x[2])
        
        # 4 - Updates Global best
        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            self.best_struc = best_struc.copy()
            self.best_con = best_con.copy()
        
        # Selection and Elitism for structures
        elite_indices = np.argsort(self.fitness_struc)[-self.elite:]
        new_pop_struc = [self.pop_struc[i].copy() for i in elite_indices]
        
        while len(new_pop_struc) < self.pop_size_struc:
            p1, p2 = self.tournament_selection(self.pop_struc, self.fitness_struc)
            child = self.crossover_struc(p1, p2)
            child = self.mutate_struc(child)
            new_pop_struc.append(child)
        
        # Selection and Elitism for controllers
        combined_con = list(zip(self.pop_con, self.fitness_con))
        combined_con.sort(key=lambda x: x[1], reverse=True)
        elite_con = [x[0] for x in combined_con[:self.elite]]
        
        new_pop_con = elite_con.copy()
        while len(new_pop_con) < self.pop_size_con:
            parent = random.choice(elite_con)
            child = self.mutate_con(parent)
            new_pop_con.append(child)
        
        new_pop_struc[0] = best_struc.copy()
        new_pop_con[0] = best_con.copy()
        
        self.pop_struc = new_pop_struc
        self.pop_con = new_pop_con
        
        return (best_struc.copy(), best_con.copy()), best_fit

    def evolve(self, generations):
        for gen in trange(generations, desc="Co-evolving"):
            (best_struc, best_con), best_fit = self.bestPairing()
            print(f"Generation {gen+1}: Best Fitness = {best_fit:.4f}")
        
        return self.best_struc, self.best_con, self.best_fitness

if __name__ == "__main__":
    co_evolver = CoEvolution()
    best_struct, best_ctrl, best_fit = co_evolver.evolve(NUM_GENERATIONS)
    
    # Visualização do melhor resultado
    print("\n=== Best Solution ===")
    print(f"Best Fitness: {best_fit:.4f}")
    print("Best Structure:")
    print(best_struct)
    
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