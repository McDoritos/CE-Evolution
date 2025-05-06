import multiprocessing
import random
import numpy as np
from utils import *
import GA_structure as ga
import ES_structure_OLD as es
import signal

def signal_handler(sig, frame):
    print("\n[INFO] Received interrupt signal. Exiting gracefully...")
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # Move the torch import here instead of global scope.
    import torch  # Now imported inside main

    multiprocessing.freeze_support()
    
    try:
        for seed in seed_list:
            print(f"Seed: {seed}")

            set_seed(seed)
            ga.run_ga()

            set_seed(seed)
            es.run_es()
    
    except KeyboardInterrupt:
        print("\n[INFO] Execution interrupted by user. Exiting gracefully.")
