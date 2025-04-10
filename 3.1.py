import random
import numpy as np
import torch
from utils import *
import GA_structure as ga
import ES_structure as es


for seed in seed_list:
    print(f"Seed: {seed}")

    set_seed(seed)
    ga.run_ga()

    set_seed(seed)
    es.run_es()

    