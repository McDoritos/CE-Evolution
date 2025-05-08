import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
from neural_controller import *
from tqdm import trange
from fixed_controllers import *
import copy

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
    def __init__(self, pop_size_struc=10, offspring_size=20 , pop_size_con=20, elite=5, mutation=0.1):
        self.pop_size_struc = pop_size_struc
        self.pop_size_con = pop_size_con
        self.mutation = mutation
        self.offspring_size = offspring_size
        self.elite = 5
        # Inicializa populações
        self.pop_struc = [self.create_random_robot() for _ in range(pop_size_struc)]
        
        self.input_range = (80,99)
        self.output_range = (4,25)

        self.pop_con = []
        for _ in range(pop_size_con):
            input_size = random.choice([80, 99]) if random.random() < 0.3 else random.randint(*self.input_range)
            output_size = random.choice([4, 25]) if random.random() < 0.3 else random.randint(*self.output_range)
            controller = NeuralController(input_size, output_size)
            self.pop_con.append({
                'weights': [np.random.randn(*p.shape) for p in controller.parameters()],
                'input_size': input_size,
                'output_size': output_size
            })
        
        self.fitness_struc = []
        self.fitness_con = []
        self.pairings = []
        
        self.best_struc = None
        self.best_con = None
        self.best_fitness = -np.inf

    def evaluate_struc(self, robot_structure, controller_weights, view=False):    
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
        
    def evaluate_con(self, controller_weights, robot_structure, view=False):
        return self.evaluate_struc(robot_structure, controller_weights, view)
    
    def create_random_robot(self):
        """Generate a valid random robot structure."""
        grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), 
                     random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
        random_robot, _ = sample_robot(grid_size)
        return random_robot
    
    def vector_to_weights(self, vector, input_size, output_size):
        temp_model = NeuralController(input_size, output_size)
        shapes = [p.shape for p in temp_model.parameters()]
        
        weights = []
        idx = 0
        for shape in shapes:
            size = np.prod(shape)
            weights.append(vector[idx:idx+size].reshape(shape))
            idx += size
        return weights
    
    def has_actuator(self, structure):
        return np.any((structure == 3) | (structure == 4))
    
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
    
    def adapt_controller(self, controller, new_input_size, new_output_size):
        old_weights = controller['weights']
        
        new_controller = NeuralController(new_input_size, new_output_size)
        new_weights = []
        
        # Para cada camada, adapta os pesos
        for i, (old_param, new_param) in enumerate(zip(old_weights, new_controller.parameters())):
            new_shape = new_param.shape
            
            if len(new_shape) == 2:  # Camadas lineares 
                if i == 0:  # Primeira camada (input)
                    min_rows = min(old_param.shape[0], new_shape[0])
                    min_cols = min(old_param.shape[1], new_shape[1])
                    new_weight = np.random.randn(*new_shape) * 0.1  # Inicialização pequena
                    new_weight[:min_rows, :min_cols] = old_param[:min_rows, :min_cols]
                else:  # Camadas ocultas/saída
                    min_size = min(old_param.shape[0], new_shape[0])
                    new_weight = np.random.randn(*new_shape) * 0.1
                    new_weight[:min_size] = old_param[:min_size]
            else:  # Bias
                min_size = min(old_param.shape[0], new_shape[0])
                new_weight = np.random.randn(*new_shape) * 0.1
                new_weight[:min_size] = old_param[:min_size]
            
            new_weights.append(new_weight)
        
        return {
            'weights': new_weights,
            'input_size': new_input_size,
            'output_size': new_output_size
        }
    
    def evolve_controllers(self):
        # 1. Seleção por elitismo + mutação forte
        sorted_cons = sorted(
            zip(self.pop_con, self.fitness_con),
            key=lambda x: x[1], 
            reverse=True
        )
        elites = [copy.deepcopy(con) for con, _ in sorted_cons[:self.elite]]

        # 2. Mutação fixa nos pesos (sigma constante)
        offspring = []
        for _ in range(self.pop_size_con - self.elite):
            parent = random.choice(elites)
            child_weights = []
            
            for weight in parent['weights']:
                # Mutação fixa (sigma=0.1) com chance de mutação (80%)
                if random.random() < 0.8:  # 80% de chance de mutar cada peso
                    noise = np.random.normal(0, 0.1, weight.shape)
                    child_weights.append(weight + noise)
                else:
                    child_weights.append(weight.copy())  # Mantém o peso original
                
            offspring.append({
                'weights': child_weights,
                'input_size': parent['input_size'],
                'output_size': parent['output_size']
            })

        # 3. Nova população = elites + offspring
        self.pop_con = elites + offspring

    def controllers_equal(self,con1, con2, tol=1e-5):
        if con1['input_size'] != con2['input_size'] or con1['output_size'] != con2['output_size']:
            return False
        for w1, w2 in zip(con1['weights'], con2['weights']):
            if not np.allclose(w1, w2, atol=tol):
                return False
        return True

    def preference_pairing(self):
        self.fitness_struc = []
        self.fitness_con = []
        self.pairings = []
        
        offsprings_struc = [self.mutate(random.choice(self.pop_struc)) for _ in range(self.offspring_size)]
        combined_population_struc = self.pop_struc + offsprings_struc

        for struc in combined_population_struc:
            
            connectivity = get_full_connectivity(struc)
            env = gym.make(SCENARIO, max_episode_steps=STEPS, body=struc, connections=connectivity)
            input_size = env.observation_space.shape[0]
            output_size = env.action_space.shape[0]
            env.close()

            best_fit = -np.inf
            best_con = None

            compatible_controllers = []
            for con in self.pop_con:
                ctrl_input = con['input_size']
                ctrl_output = con['output_size']

                if ctrl_input == input_size and ctrl_output == output_size:
                    print("Compatible controller found")
                    compatible_controllers.append(con)

            if compatible_controllers:
                if self.best_con == None:
                    partner = random.choice(compatible_controllers)
                else:
                    if not any(self.controllers_equal(self.best_con, con) for con in compatible_controllers):
                        partner = random.choice(compatible_controllers)
                    else:
                        partner = self.best_con
            else:
                print("No compatible controller, adapting...")
                partner = self.best_con if self.best_con is not None else random.choice(self.pop_con)
                print("Size of controller before:",partner['input_size'],partner['output_size'])
                adapted_con = self.adapt_controller(partner, input_size, output_size)
                self.pop_con.append(adapted_con)
                partner = adapted_con
                print("Size of controller after adapting:",partner['input_size'],partner['output_size'])
            
            fit = self.evaluate_struc(struc, partner['weights'])
            self.fitness_struc.append(fit)
            self.fitness_con.append(fit)
            self.pairings.append((struc, partner, fit))

        print("Fitness das estruturas: \n",self.fitness_struc)

        best_struc, best_con, best_fit = max(self.pairings, key=lambda x: x[2])
        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            self.best_struc = best_struc.copy()
            self.best_con = best_con.copy()

        self.evolve_controllers()

        sorted_population = [x for _, x in sorted(zip(self.fitness_struc, combined_population_struc), key=lambda pair: pair[0], reverse=True)]
        self.pop_struc = sorted_population[:POPULATION_SIZE]

        return (best_struc.copy(), best_con.copy()), best_fit
    

    def evolve(self, generations):
        for gen in trange(generations, desc="Co-evolving"):
            (best_struc, best_con), best_fit = self.preference_pairing()
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