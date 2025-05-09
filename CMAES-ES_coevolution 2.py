from datetime import datetime
import multiprocessing
import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
from neural_controller import *
from tqdm import trange
from fixed_controllers import *
import cma
import copy
import utils

STEPS = 500
MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)
VOXEL_TYPES = [0, 1, 2, 3, 4]  # 0: vazio, 1: rígido, 2: macio, 3: atuador+, 4: atuador-
NUM_GENERATIONS = 100

class CoEvolution:
    def __init__(self, scenario, seed, seed_folder, pop_size_struc=10, offspring_size=20, pop_size_con=20, elite=5, mutation=0.1):
        utils.set_seed(seed)
        self.STRUCTURE_POP_SIZE = pop_size_struc
        self.CONTROLLER_POP_SIZE = pop_size_con
        self.mutation = mutation
        self.OFFSPRING_SIZE = offspring_size
        self.seed_folder = seed_folder
        self.scenario = scenario
        # Best structure from 3.1 scenario
        robot_structure = np.array([ 
            [3,3,3,3,3],
            [3,3,3,3,3],            
            [3,3,3,3,3],            
            [3,3,3,3,3],
            [3,3,3,3,3]
            ])
        # Inicializa populações
        self.struct_population = [self.create_random_robot() for _ in range(pop_size_struc)]

        env = gym.make(scenario, body=robot_structure)
        self.max_input_size = 120
        self.max_output_size = env.action_space.shape[0]

        self.brain_template = NeuralController(self.max_input_size, self.max_output_size)
        self.controller_population = [[np.random.randn(*p.shape) for p in self.brain_template.parameters()] 
                    for _ in range(pop_size_con)]
        env.close()

        self.pairings = [] # (estrutura, controlador, fitness)
        
        self.best_structure = None
        self.best_controller = None
        self.best_fitness = -np.inf
        self.init_cma_es(seed)

    def init_cma_es(self, seed):
        example_weights = [p.data.numpy() for p in self.brain_template.parameters()]
        self.weight_shapes = [w.shape for w in example_weights]
        initial_params = np.concatenate([w.flatten() for w in example_weights])
        
        self.cma_es = cma.CMAEvolutionStrategy(
            initial_params,
            0.5,
            {'popsize': self.CONTROLLER_POP_SIZE, 'seed': seed}
        )

    def get_actuator_indices(self, structure):
        flat_structure = structure.flatten()
        return np.flatnonzero((flat_structure == 3) | (flat_structure == 4))

    def adapt_controller(self, controller, new_input_size, new_output_size):
        old_weights = controller
        
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
        
        return new_weights
    


    def evaluate(self, robot_structure, controller_weights, scenario, view=False):    
        connectivity = get_full_connectivity(robot_structure)
        env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]



        # if input_size > self.max_input_size:
        #     raise ValueError(f"Input size {input_size} exceeds maximum size {self.max_input_size}.")
        
                    
        # actuator_indices = self.get_actuator_indices(robot_structure)
        # W1, b1, W2, b2 = controller_weights
        # assert W1.shape[1] >= input_size, f"Controller input too small: {W1.shape[1]} vs {input_size}"

        # W1_new = W1[:, :input_size]
        # W2_new = W2[actuator_indices, :]
        # b2_new = b2[actuator_indices]

        # controller_weights = [W1_new, b1, W2_new, b2_new]
        controller_weights = self.adapt_controller(controller_weights, input_size, output_size)

        brain = NeuralController(input_size, output_size)
        set_weights(brain, controller_weights)
        
        # evaluate
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

    def mutate(self, parent, max_attempts=5):
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

        return parent

    def best_pairing(self):
        self.structure_fitness = []
        self.controller_fitness = []
        self.pairings = []
        
        # evolve structure
        offsprings = [self.mutate(random.choice(self.struct_population)) for _ in range(self.OFFSPRING_SIZE)]
        structure_combined_population = self.struct_population + offsprings

        best_controller = self.best_controller if self.best_controller is not None else random.choice(self.controller_population)

        for structure in structure_combined_population:
            fit = self.evaluate(structure, best_controller, self.scenario)
            self.structure_fitness.append(fit)
            self.pairings.append((structure, best_controller, fit))


        solutions = self.cma_es.ask()

        if self.structure_fitness:
            best_structure = structure_combined_population[np.argmax(np.array(self.structure_fitness))]
        else:
            best_structure = random.choice(structure_combined_population)
        
        for controller in solutions:
            weights = self.vector_to_weights(controller)
            fit = self.evaluate(best_structure, weights, self.scenario)
            self.controller_fitness.append(-fit)  # CMA-ES minimiza
            self.pairings.append((best_structure, weights, fit))
        
        self.cma_es.tell(solutions, self.controller_fitness)
        self.controller_population = [self.vector_to_weights(x) for x in solutions]
        self.controller_fitness = [-f for f in self.controller_fitness]

        # Encontra o melhor par
        best_structure, best_controller, generation_best_fitness = max(self.pairings, key=lambda x: x[2])
        
        # Atualiza o melhor global
        if generation_best_fitness > self.best_fitness:
            self.best_fitness = generation_best_fitness
            self.best_structure = best_structure.copy()
            self.best_controller = best_controller.copy()
        
        sorted_population = [x for _, x in sorted(zip(self.structure_fitness, structure_combined_population), key=lambda pair: pair[0], reverse=True)]
        self.struct_population = sorted_population[:self.STRUCTURE_POP_SIZE]
        

        return best_structure.copy(), best_controller.copy(), generation_best_fitness

    def evolve(self, generations):
        for gen in trange(generations, desc="Co-evolving"):
            best_structure, best_controller, best_fitness = self.best_pairing()

            gif_folder = os.path.join(self.seed_folder, f"gif.gif")
            utils.create_gifs(best_structure, scenario=self.scenario, controller=best_controller, steps=STEPS, filename=gif_folder)
            csv_filename = os.path.join(self.seed_folder, f"gen_{gen}.csv")
            utils.save_pairing(self.pairings, csv_filename)

            print(f"Generation {gen+1}: Best Fitness (global) = {self.best_fitness:.4f} ; Best Fitness (gen) = {best_fitness:.4f}")
        
        return self.best_structure, self.best_controller, self.best_fitness
    
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

    utils.create_gif(best_structure, scenario=scenario, controller=best_controller, steps=STEPS)    

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

    # scenario = "GapJumper-v0"
    # seed = 42
    # run_seed(seed, scenario)