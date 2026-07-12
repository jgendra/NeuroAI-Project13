"""
mante_config.py
Central configuration for Mante et al. (2013) dataset generation.
"""

import numpy as np

# Total trial duration = 1150ms at dt=10ms means exactly 115 timesteps.
TIMING = {
    "fixation": 300,
    "stimulus": 750,
    "delay": 0,
    "decision": 100,
}
DT = 10
TOTAL_TIMESTEPS = sum(TIMING.values()) // DT

# Coherence distributions
# NeuroGym expects positive coherences and applies the +/- internally based on ground truth.

# Mante's raw fractions: 0.009, 0.036, 0.15
# NeuroGym requires them scaled by 100:
MANTE_TEST_COHS = [0.9, 3.6, 15.0]

# Mante's uniform distribution bound: 0.1875
# Scaled by 100: 18.75
UNIFORM_COHS = np.linspace(0.0, 18.75, 50).tolist()

CONFIG = {
    "task": "ContextDecisionMaking-v0",
    "dt": DT,
    "sigma": 1.0,  # Noise standard deviation
    "seq_len": TOTAL_TIMESTEPS,
    "timing": TIMING,
    
    # Dataset sizes
    "splits": {
        "train": 160000,
        "val": 2000,           # Used for early stopping
        "test_uniform": 2000,  # Psychometric curve testing
        "test_mante": 2000     # Specific Mante coherence testing
    },
    
    "seed_train": 42,
    "seed_val": 43,
    "output_dir": "data/"
}
