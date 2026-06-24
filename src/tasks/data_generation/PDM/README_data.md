# PDM Dataset — Generation & Validation

Dataset pipeline for **Project 13: Information Decomposition in Task-Trained RNNs**.  
Generates and validates the `PerceptualDecisionMaking-v0` dataset used to train the Elman RNN and run Gaussian analytic PID analysis.

---

## Files

| File | Purpose |
|---|---|
| `generate_pdm_dataset.py` | Generate train / val / test `.npz` splits from NeuroGym |
| `test_dataset.py` | Validate generated splits before training |

---

## Requirements

```bash
pip install neurogym torch numpy
```

Tested with **NeuroGym 2.3.1**, PyTorch ≥ 2.0, Python ≥ 3.10.

---

## Quick start

```bash
# 1. Generate full dataset
python generate_pdm_dataset.py

# 2. Validate before training
python test_dataset.py
```

---

## `generate_pdm_dataset.py`

### What it does

Generates three reproducible splits of `PerceptualDecisionMaking-v0` trials and saves them as compressed `.npz` files. Each trial is a complete fixed-length sequence covering fixation → stimulus → decision periods.

### Output files

Saved to `data/pdm/` by default:

```
data/pdm/
    train.npz
    val.npz
    test.npz
    config.npz      ← full config saved for reproducibility
```

Each `.npz` contains four arrays:

| Array | Shape | Dtype | Description |
|---|---|---|---|
| `observations` | `(n_trials, seq_len, 3)` | float32 | RNN input at every timestep |
| `labels` | `(n_trials, seq_len)` | int64 | Ground-truth action per timestep |
| `coherences` | `(n_trials,)` | float32 | Signed stimulus coherence — PID target variable |
| `trial_periods` | `(n_trials, seq_len)` | int8 | Period label per timestep |

**Period labels:** `0` = fixation · `1` = stimulus · `2` = decision

**Label values:** `0` = fixate · `1` = choose ch1 · `2` = choose ch2

**Signed coherence:** `+coh` when ground truth is choice 1, `−coh` when choice 2. Used as the continuous PID target variable rather than the binary label.

### Observation vector (ob_size = 3)

| Channel | Content | Active during |
|---|---|---|
| 0 | Fixation signal (1.0 or 0.0) | Fixation period |
| 1 | Noisy stimulus toward choice 1: `+coh/100 + N(0, σ)` | Stimulus period |
| 2 | Noisy stimulus toward choice 2: `−coh/100 + N(0, σ)` | Stimulus period |

### Trial structure at dt = 20 ms

```
Period      Duration    Steps
─────────────────────────────
Fixation    100 ms       5
Stimulus    2000 ms    100
Decision    100 ms       5
─────────────────────────────
Total       2200 ms    110
```

`seq_len = 120` gives 10 steps of padding beyond the 110-step trial, ensuring the decision period is never truncated.

### Default configuration

| Parameter | Default | Description |
|---|---|---|
| `dt` | 20 ms | Timestep — overrides NeuroGym default of 100 ms |
| `cohs` | [0, 6.4, 12.8, 25.6, 51.2] | Coherence levels (Britten 1992 standard set) |
| `sigma` | 1.0 | Input noise std per channel per timestep |
| `seq_len` | 120 | Timesteps per trial (must be ≥ 110) |
| `n_train` | 10 000 | Training trials |
| `n_val` | 1 000 | Validation trials |
| `n_test` | 1 000 | Test trials |
| `seed` | 42 | Master seed (train/val/test get derived seeds) |
| `output_dir` | `data/pdm` | Output directory |

### CLI arguments

```bash
python generate_pdm_dataset.py \
    --n_train 10000 \
    --n_val   1000  \
    --n_test  1000  \
    --seq_len 120   \
    --dt      20    \
    --sigma   1.0   \
    --seed    42    \
    --output_dir data/pdm \
    --print_every 500
```

### Loading the data

```python
import numpy as np

train = np.load('data/pdm/train.npz')

obs     = train['observations']   # (10000, 120, 3)  float32
labels  = train['labels']         # (10000, 120)     int64
cohs    = train['coherences']     # (10000,)         float32
periods = train['trial_periods']  # (10000, 120)     int8

# Extract stimulus-end snapshot for PID (last timestep of stimulus period)
stim_end_idx = np.array([np.where(p == 1)[0][-1] for p in periods])
```

### Using with PyTorch DataLoader

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

obs_t  = torch.tensor(obs)
labs_t = torch.tensor(labels)
pers_t = torch.tensor(periods)

dataset    = TensorDataset(obs_t, labs_t, pers_t)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for inputs, targets, pds in dataloader:
    # Decision-period masked loss
    mask        = (pds == 2)
    outputs     = model(inputs)             # (B, T, act_size)
    masked_loss = criterion(
        outputs[mask],                      # (n_dec_steps, act_size)
        targets[mask]                       # (n_dec_steps,)
    )
```

---

## `test_dataset.py`

### What it does

Runs four levels of checks on the generated `.npz` files to catch any issues before training.

### Checks

**1. File & shape integrity**
- All three split files exist
- All four required keys present
- Shapes consistent across arrays
- Correct dtypes (`float32` for observations, `int64` for labels)

**2. Value sanity**
- No NaN or Inf in observations
- Label values in valid range (0–2)
- Signed coherence has both positive and negative values
- Roughly balanced choice1 / choice2 trial counts
- Period labels `[0, 1, 2]` all present
- All trials start in fixation period
- Decision period present in every trial

**3. Training pipeline simulation**
- Instantiates an Elman RNN (100 hidden units, tanh) matching the project spec
- Runs a real forward + backward pass through one batch
- Checks all parameters received non-zero gradients
- Tests decision-period masked cross-entropy loss specifically

**4. PID-readiness**
- Finds the modal stimulus-end timestep across test trials
- Warns if stimulus timing varies too much across trials
- Checks coherence has sufficient variance to serve as a PID target
- Confirms hidden-state snapshot shape will be correct for Gaussian analytic PID

### Usage

```bash
python test_dataset.py                        # default: data/pdm/
python test_dataset.py --data_dir my/path
```

### Expected output (all passing)

```
=======================================================
  PDM Dataset Validation
  Directory: data/pdm
=======================================================
  1. File & shape integrity
  ✓ Loaded train.npz
  ✓   Keys present: ['coherences', 'labels', 'observations', 'trial_periods']
  ✓   Shapes — obs:(10000, 120, 3) ...
  ✓   Dtypes — obs:float32  labs:int64 ...

  2. Value sanity
  ✓ No NaN/Inf in observations
  ✓ Label values: [0 1 2]
  ✓ Signed coherences present (both + and −)
  ✓ Period values: [0 1 2]
  ✓ Decision period present in all trials

  3. Training-pipeline simulation
  ✓ Forward + backward pass completed
  ✓ All parameters received non-zero gradients
  ✓ Decision-masked loss: X.XXXX

  4. PID-readiness
  ✓ Modal stimulus-end timestep: t=104
  ✓ Data is PID-ready

  All checks passed — dataset is training-ready.
=======================================================
```

---

## Known issues

**NeuroGym 2.3.1 deprecation warnings** — accessing `env.ob`, `env.gt`, `env.trial` directly raises gymnasium wrapper deprecation warnings. These are harmless. The scripts use `env.unwrapped.*` to suppress them where possible.

**Gymnasium render_modes warning** — `UserWarning: WARN: The environment creator metadata doesn't include render_modes`. Harmless, can be ignored.

---

## Reproducibility

Each split uses a deterministic seed derived from the master `--seed` argument:

```python
rng        = np.random.RandomState(seed)
train_seed = rng.randint(0, 2**31)
val_seed   = rng.randint(0, 2**31)
test_seed  = rng.randint(0, 2**31)
```

The full config is saved to `data/pdm/config.npz` alongside the data so the exact generation parameters are always recoverable.

---

## Project context

These scripts are part of a controlled study comparing Partial Information Decomposition (PID) profiles of Elman RNNs trained on tasks with different integration demands (Project 13, NeuroAI & ML course, TU Munich, SS 2026). The test split is reserved exclusively for PID analysis and is never used during training.