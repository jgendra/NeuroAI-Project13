# Mante 2013 Dataset Pipeline — Generation & Loading

Dataset pipeline for Project 13: Information Decomposition in Task-Trained RNNs.
This pipeline replaces the standard PerceptualDecisionMaking-v0 to perfectly replicate the biological constraints and signal-to-noise ratios of Mante et al. (2013).

## Files

- `mante_config.py`: Central configuration (timings, coherence scaling, dataset splits).
- `data_generator.py`: Generates .npz data splits for both Context and Perceptual modes.
- `data_loader.py`: PyTorch DataLoader wrapper with built-in temporal subsampling.

## The Two Modes (Isolating Integration Demand)

To prove that Synergy differences are due to cognitive integration demand and not simply network dimensionality, both tasks use the exact same 7-channel `ContextDecisionMaking-v0` environment:
1. `--mode context` **(High Integration):** Both sensory streams (Modality 1 and 2) are active. The network uses the explicitly signaled context cue to route the correct stream and ignore the noisy distractor.
2. `--mode perceptual` **(Low Integration - Masked):** The context cue is clamped to Modality 1, and the Modality 2 sensory stream is mathematically zeroed out ("dead wires"). The network simply accumulates the active stream without any cognitive conflict or dynamic routing.

## Biological Constraints
### 1. Coherence Scaling

Mante et al. define coherences as fractional differences (e.g., c = 0.15). NeuroGym's internal ring-representation math divides inputs by 200. To recreate the exact signal-to-noise ratio from the 2013 paper, we multiply Mante's fractions by 100 in our config (e.g., `0.15` becomes `15.0`).
### 2. The Time Constant (tau) & Subsampling

Mante et al. used a continuous-time resolution of `dt=1ms` with a total trial duration of 750ms.

- **CTRNNs** handle 750 timesteps natively via their leak parameter.
- **Elman RNNs** suffer from vanishing gradients over 750 steps.

**The Fix:** `data_loader.py` includes a `subsample_step` parameter. By loading the data with `subsample_step=10`, you compress the 750-step array into 75 steps, perfectly mimicking a 10ms biological integration window for the discrete Elman RNN.

## Quick Start
### 1. Generate the Data

(Warning: Generating the full 160k trial datasets takes a few minutes and ~3.5GB of disk space. Do not push the .npz files to GitHub).
```Bash

# Generate the High Integration target
python data_generator.py --mode context

# Generate the Low Integration control
python data_generator.py --mode perceptual
```

### 2. Generate a Tiny Test Set (For Debugging)
```Bash

python data_generator.py --mode context --dt 1 --n_train 500 --n_val 100 --n_test_uni 100 --n_test_mante 0 --output_dir data/tiny_test
```

### 3. Load into PyTorch
```Python

from data_loader import load_mante_data

# Elman RNN loader (subsampled to 75 steps to prevent gradient vanishing)
elman_loader = load_mante_data('data/mante_style/context/train.npz', batch_size=64, subsample_step=10)

# CTRNN loader (full 750 step biological resolution)
ctrnn_loader = load_mante_data('data/mante_style/context/train.npz', batch_size=64, subsample_step=1)
```