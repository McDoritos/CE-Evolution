import random
import numpy as np

import gymnasium as gym
from evogym.envs import *
from evogym import sample_robot

MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)

class CoEvolution:
    def __init__(self, pop_size_structures=3, offspring_size=3, pop_size_controllers=20, pairing_elite=5, mutation_rate=0.2):
        self.pop_size_structures = pop_size_structures
        self.pop_size_controllers = pop_size_controllers
        self.mutation_rate = mutation_rate
        self.offspring_size = offspring_size
        self.pairing_elite = pairing_elite
        
        # Inicializa população de estruturas
        self.structures_pop = [self.create_random_robot() for _ in range(pop_size_structures)]
        
        self.structures_fitness = []
        self.controller_fitness = []
        self.pairings = []
        
        self.best_structure = None
        self.best_controller = None
        self.best_fitness = -np.inf

        # self.fitness_history = []  # Histórico para mutação adaptativa

    def create_random_robot(self):
        """Generate a valid random robot structure."""
        grid_size = (random
                     .randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), 
                     random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
        random_robot, _ = sample_robot(grid_size)
        return random_robot