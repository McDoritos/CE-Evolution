import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
from neural_controller import *
from tqdm import trange
from fixed_controllers import *
import cma

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
    def __init__(self, pop_size_struc=20, pop_size_con=20, elite=5, mutation=0.05, offspring_size_con=20):
        self.pop_size_struc = pop_size_struc
        self.pop_size_con = pop_size_con
        self.offspring_size_con = offspring_size_con
        self.elite = elite
        self.mutation = mutation
        
        # Best structure from 3.1 scenario
        robot_struc = np.array([ 
            [1,3,3,3,1],
            [1,1,3,2,4],
            [4,4,3,3,3],
            [2,4,0,0,1],
            [4,4,0,0,1]
            ])
        # Inicializa populações
        self.pop_struc = [self.create_random_robot() for _ in range(pop_size_struc)]

        env = gym.make(SCENARIO, body=robot_struc)  # Usa a primeira estrutura como exemplo
        self.FIXED_INPUT_SIZE = env.observation_space.shape[0]
        self.FIXED_OUTPUT_SIZE = env.action_space.shape[0]
        print(self.FIXED_INPUT_SIZE,self.FIXED_OUTPUT_SIZE)

        self.brain_template = NeuralController(self.FIXED_INPUT_SIZE, self.FIXED_OUTPUT_SIZE)
        self.pop_con = [[np.random.randn(*p.shape) for p in self.brain_template.parameters()] 
                    for _ in range(pop_size_con)]
        env.close()
        
        self.fitness_struc = []
        self.fitness_con = []
        self.pairings = []
        
        self.best_struc = None
        self.best_con = None
        self.best_fitness = -np.inf
        self.init_cma_es()

    def init_cma_es(self):
        example_weights = [p.data.numpy() for p in self.brain_template.parameters()]
        self.weight_shapes = [w.shape for w in example_weights]
        initial_params = np.concatenate([w.flatten() for w in example_weights])
        
        self.cma_es = cma.CMAEvolutionStrategy(
            initial_params,
            0.5,  # Sigma inicial (ajuste conforme necessário)
            {'popsize': self.pop_size_con, 'seed': SEED}
        )

    def evaluate_struc(self, robot_structure, controller_weights, view=False):    
        try:
            connectivity = get_full_connectivity(robot_structure)
            env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
            input_size = env.observation_space.shape[0]
            output_size = env.action_space.shape[0]

            if input_size != self.FIXED_INPUT_SIZE or output_size != self.FIXED_OUTPUT_SIZE:
                env.close()
                penalty = -abs(input_size - self.FIXED_INPUT_SIZE) - abs(output_size - self.FIXED_OUTPUT_SIZE)
                return penalty

            brain = NeuralController(self.FIXED_INPUT_SIZE, self.FIXED_OUTPUT_SIZE)
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
            print(f"Error in evaluation: {str(e)}")
            return -1000
        
    def evaluate_con(self, controller_weights, robot_structure, view=False):
        return self.evaluate_struc(robot_structure, controller_weights, view)
    
    def create_random_robot(self):
        """Generate a valid random robot structure."""
        grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), 
                     random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
        random_robot, _ = sample_robot(grid_size)
        return random_robot
    
    def vector_to_weights(self, vector):
        weights = []
        idx = 0
        for shape in self.weight_shapes:
            size = np.prod(shape)
            weights.append(vector[idx:idx+size].reshape(shape))
            idx += size
        return weights
    
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
    
    def best_pairing(self):
        self.fitness_struc = []
        self.fitness_con = []
        self.pairings = []
        
        # 1 - Avalia estruturas com o melhor controlador atual
        best_con = self.best_con if self.best_con is not None else random.choice(self.pop_con)
        for structure in self.pop_struc:
            fit = self.evaluate_struc(structure, best_con)
            self.fitness_struc.append(fit)
            self.pairings.append((structure, best_con, fit))
        
        # Gera e avalia nova população de controladores
        solutions = self.cma_es.ask()
        best_struc = self.pop_struc[np.argmax(self.fitness_struc)] if self.fitness_struc else random.choice(self.pop_struc)
        
        fitness = []
        for x in solutions:
            weights = self.vector_to_weights(x)
            fit = self.evaluate_con(weights, best_struc)
            fitness.append(-fit)  # CMA-ES minimiza
            self.pairings.append((best_struc, weights, fit))
        
        self.cma_es.tell(solutions, fitness)
        self.pop_con = [self.vector_to_weights(x) for x in solutions]
        self.fitness_con = [-f for f in fitness]

        # 3 - Encontra o melhor par
        best_struc, best_con, best_fit = max(self.pairings, key=lambda x: x[2])
        
        # 4 - Atualiza o melhor global
        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            self.best_struc = best_struc.copy()
            self.best_con = best_con.copy()
        
        # 5 - Seleção para estruturas (mantido original)
        elite_indices = np.argsort(self.fitness_struc)[-self.elite:]
        new_pop_struc = [self.pop_struc[i].copy() for i in elite_indices]
        
        while len(new_pop_struc) < self.pop_size_struc:
            p1, p2 = self.tournament_selection(self.pop_struc, self.fitness_struc)
            child = self.crossover_struc(p1, p2)
            child = self.mutate_struc(child)
            new_pop_struc.append(child)
        
        # 6 - Elitismo para controladores (agora desnecessário, pois o CMA-ES já faz seleção)
        # Mantemos apenas para garantir a população ter tamanho correto
        self.pop_struc = new_pop_struc.copy()
        
        return (best_struc.copy(), best_con.copy()), best_fit
    
    def random_pairing(self):
        self.fitness_struc = []
        self.fitness_con = []
        self.pairings = []
        
        # 1 - Avalia estruturas com o melhor controlador atual
        for structure in self.pop_struc:
            partner = random.choice(self.pop_con)
            fit = self.evaluate_struc(structure, partner)
            self.fitness_struc.append(fit)
            self.pairings.append((structure, partner, fit))
        
        # Gera e avalia nova população de controladores
        solutions = self.cma_es.ask()
        
        fitness = []
        for x in solutions:
            partner = random.choice(self.pop_struc)
            weights = self.vector_to_weights(x)
            fit = self.evaluate_con(weights, partner)
            fitness.append(-fit)  # CMA-ES minimiza
            self.pairings.append((partner, weights, fit))
        
        self.cma_es.tell(solutions, fitness)
        self.pop_con = [self.vector_to_weights(x) for x in solutions]
        self.fitness_con = [-f for f in fitness]

        # 3 - Encontra o melhor par
        best_struc, best_con, best_fit = max(self.pairings, key=lambda x: x[2])
        
        # 4 - Atualiza o melhor global

        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            self.best_struc = best_struc.copy()
            self.best_con = best_con.copy()
        
        # 5 - Seleção para estruturas (mantido original)
        elite_indices = np.argsort(self.fitness_struc)[-self.elite:]
        new_pop_struc = [self.pop_struc[i].copy() for i in elite_indices]
        
        while len(new_pop_struc) < self.pop_size_struc:
            p1, p2 = self.tournament_selection(self.pop_struc, self.fitness_struc)
            child = self.crossover_struc(p1, p2)
            child = self.mutate_struc(child)
            new_pop_struc.append(child)
        
        # 6 - Elitismo para controladores (agora desnecessário, pois o CMA-ES já faz seleção)
        # Mantemos apenas para garantir a população ter tamanho correto
        self.pop_struc = new_pop_struc.copy()
        
        return (best_struc.copy(), best_con.copy()), best_fit

    def evolve(self, generations):
        for gen in trange(generations, desc="Co-evolving"):
            (best_struc, best_con), best_fit = self.random_pairing()
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