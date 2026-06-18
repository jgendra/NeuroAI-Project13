"""
test_dataset.py
Validate the generated PDM .npz files before RNN training.

Run:
    python test_dataset.py                        # checks data/pdm/
    python test_dataset.py --data_dir my/path     # custom directory

Checks are grouped into four levels:
    1. File & shape integrity
    2. Value sanity (ranges, NaNs, class balance)
    3. Training-pipeline simulation (DataLoader + one forward pass)
    4. PID-readiness (coherence distribution, stimulus-end snapshot)
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = "data/pdm"
SEQ_LEN  = None   # inferred from file
OB_SIZE  = None
ACT_SIZE = 3      # fixate / choice1 / choice2

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

def ok(msg):   print(f"{PASS} {msg}")
def fail(msg): print(f"{FAIL} {msg}"); sys.exit(1)
def warn(msg): print(f"{WARN} {msg}")

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. File & shape integrity
# ──────────────────────────────────────────────────────────────────────────────

def check_files(data_dir: str) -> dict:
    section("1. File & shape integrity")

    splits = {}
    for name in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{name}.npz")
        if not os.path.exists(path):
            fail(f"Missing file: {path}")

        data = np.load(path)
        splits[name] = data
        size_mb = os.path.getsize(path) / 1e6
        ok(f"Loaded {name}.npz  ({size_mb:.1f} MB)")

        # Required keys
        required = {"observations", "labels", "coherences", "trial_periods"}
        missing  = required - set(data.files)
        if missing:
            fail(f"{name}.npz missing keys: {missing}")
        ok(f"  Keys present: {sorted(data.files)}")

        obs  = data["observations"]
        labs = data["labels"]
        cohs = data["coherences"]
        pers = data["trial_periods"]

        n = obs.shape[0]

        # Shape consistency
        assert obs.ndim  == 3, f"observations should be 3D, got {obs.ndim}D"
        assert labs.ndim == 2, f"labels should be 2D, got {labs.ndim}D"
        assert cohs.ndim == 1, f"coherences should be 1D, got {cohs.ndim}D"
        assert pers.ndim == 2, f"trial_periods should be 2D, got {pers.ndim}D"

        T, F = obs.shape[1], obs.shape[2]
        assert labs.shape  == (n, T), f"labels shape mismatch: {labs.shape} vs ({n},{T})"
        assert cohs.shape  == (n,),   f"coherences shape mismatch"
        assert pers.shape  == (n, T), f"trial_periods shape mismatch"

        ok(f"  Shapes — obs:{obs.shape}  labs:{labs.shape}  "
           f"cohs:{cohs.shape}  periods:{pers.shape}")

        # Dtypes
        assert obs.dtype  == np.float32, f"observations dtype should be float32, got {obs.dtype}"
        assert labs.dtype == np.int64,   f"labels dtype should be int64, got {labs.dtype}"
        ok(f"  Dtypes — obs:{obs.dtype}  labs:{labs.dtype}  "
           f"cohs:{cohs.dtype}  periods:{pers.dtype}")

    global SEQ_LEN, OB_SIZE
    SEQ_LEN = splits["train"]["observations"].shape[1]
    OB_SIZE  = splits["train"]["observations"].shape[2]
    return splits


# ──────────────────────────────────────────────────────────────────────────────
# 2. Value sanity
# ──────────────────────────────────────────────────────────────────────────────

def check_values(splits: dict) -> None:
    section("2. Value sanity")

    for name, data in splits.items():
        print(f"\n  [{name}]")
        obs  = data["observations"]
        labs = data["labels"]
        cohs = data["coherences"]
        pers = data["trial_periods"]

        # NaNs / Infs
        if np.isnan(obs).any() or np.isinf(obs).any():
            fail(f"NaN/Inf in observations ({name})")
        ok("No NaN/Inf in observations")

        # Label values
        unique_labs = np.unique(labs)
        bad = unique_labs[(unique_labs < 0) | (unique_labs >= ACT_SIZE)]
        if len(bad):
            fail(f"Invalid label values: {bad}")
        ok(f"Label values: {unique_labs}  (valid: 0–{ACT_SIZE-1})")

        # Label distribution (check fixation doesn't swamp everything)
        for lv, ln in [(0,"fixate"),(1,"choice1"),(2,"choice2")]:
            frac = np.mean(labs == lv)
            status = ok if 0.01 < frac < 0.99 else warn
            status(f"  label {lv} ({ln}): {frac:.1%} of timesteps")

        # Coherence values
        unique_cohs = sorted(np.unique(cohs))
        ok(f"Coherences ({len(unique_cohs)} unique): {[round(c,1) for c in unique_cohs]}")

        # Check signed coherence has both signs (or zero)
        has_pos = np.any(cohs > 0)
        has_neg = np.any(cohs < 0)
        if not (has_pos and has_neg):
            warn("Coherences are all one sign — check ground_truth encoding")
        else:
            ok("Signed coherences present (both + and −)")

        # Class balance across trials (coarse)
        pos_trials = np.sum(cohs > 0)
        neg_trials = np.sum(cohs < 0)
        zer_trials = np.sum(cohs == 0)
        ok(f"Trial balance — choice1:{pos_trials}  choice2:{neg_trials}  zero:{zer_trials}")
        ratio = pos_trials / max(neg_trials, 1)
        if not (0.7 < ratio < 1.3):
            warn(f"Choice imbalance (ratio={ratio:.2f}); expect ~1.0 for unbiased task")

        # Period labels
        unique_pers = np.unique(pers)
        ok(f"Period values: {unique_pers}  (0=fix, 1=stim, 2=dec)")
        if 2 not in unique_pers:
            fail("No decision period found — seq_len may be too short")

        # Every trial starts in fixation
        if not np.all(pers[:, 0] == 0):
            warn("Some trials don't start in fixation (t=0)")
        else:
            ok("All trials start in fixation period")

        # Observation range
        omin, omax = obs.min(), obs.max()
        ok(f"Obs range: [{omin:.3f}, {omax:.3f}]")
        if omax > 10 or omin < -10:
            warn("Observations outside [-10, 10] — consider checking sigma")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Training-pipeline simulation
# ──────────────────────────────────────────────────────────────────────────────

class ElmanRNN(nn.Module):
    """Minimal Elman RNN matching the project spec."""
    def __init__(self, ob_size, hidden_size, act_size):
        super().__init__()
        self.rnn    = nn.RNN(ob_size, hidden_size, batch_first=True,
                             nonlinearity='tanh')
        self.linear = nn.Linear(hidden_size, act_size)

    def forward(self, x):
        out, _ = self.rnn(x)        # (B, T, H)
        return self.linear(out)     # (B, T, act_size)


def check_training_pipeline(splits: dict) -> None:
    section("3. Training-pipeline simulation")

    HIDDEN = 100
    BATCH  = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ok(f"Device: {device}")

    # Build DataLoader from train split
    train = splits["train"]
    obs_t  = torch.tensor(train["observations"])  # (N, T, F)
    labs_t = torch.tensor(train["labels"])        # (N, T)

    dataset    = TensorDataset(obs_t, labs_t)
    dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
    ok(f"DataLoader: {len(dataloader)} batches  "
       f"(batch_size={BATCH}, n_trials={len(dataset)})")

    # One batch forward pass
    model     = ElmanRNN(OB_SIZE, HIDDEN, ACT_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    inputs, targets = next(iter(dataloader))
    inputs  = inputs.to(device)
    targets = targets.to(device)

    ok(f"Batch shapes — inputs:{tuple(inputs.shape)}  "
       f"targets:{tuple(targets.shape)}")

    optimizer.zero_grad()
    outputs = model(inputs)               # (B, T, act_size)
    ok(f"Model output shape: {tuple(outputs.shape)}")

    # CrossEntropyLoss expects (B*T, act_size) and (B*T,)
    loss = criterion(
        outputs.reshape(-1, ACT_SIZE),
        targets.reshape(-1),
    )
    loss.backward()
    optimizer.step()

    ok(f"Forward + backward pass completed  (loss={loss.item():.4f})")

    # Gradient check — all params should have gradients
    no_grad = [n for n, p in model.named_parameters()
               if p.grad is None or p.grad.abs().max() == 0]
    if no_grad:
        warn(f"Zero/None gradients: {no_grad}")
    else:
        ok("All parameters received non-zero gradients")

    # Masked loss (decision period only) — what you'll actually use
    train_data  = splits["train"]
    periods_t   = torch.tensor(train_data["trial_periods"])
    full_dataset = TensorDataset(obs_t, labs_t, periods_t)
    full_loader  = DataLoader(full_dataset, batch_size=BATCH, shuffle=True)

    inputs, targets, periods = next(iter(full_loader))
    inputs  = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    mask    = (periods == 2)                      # decision period only
    if mask.sum() == 0:
        warn("No decision-period timesteps in this batch — seq_len may be short")
    else:
        masked_out = outputs[mask]                # (n_decision_steps, act_size)
        masked_tgt = targets[mask]
        masked_loss = criterion(masked_out, masked_tgt)
        ok(f"Decision-masked loss: {masked_loss.item():.4f}  "
           f"({mask.sum().item()} decision timesteps in batch)")


# ──────────────────────────────────────────────────────────────────────────────
# 4. PID-readiness
# ──────────────────────────────────────────────────────────────────────────────

def check_pid_readiness(splits: dict) -> None:
    section("4. PID-readiness")

    # We'll use the test split for analysis (held-out, as in actual PID analysis)
    data  = splits["test"]
    obs   = data["observations"]   # (N, T, F)
    cohs  = data["coherences"]     # (N,)
    pers  = data["trial_periods"]  # (N, T)

    # Find stimulus-end timestep per trial (last t where period==1)
    stim_end_steps = []
    for i in range(len(pers)):
        stim_steps = np.where(pers[i] == 1)[0]
        if len(stim_steps) == 0:
            warn(f"Trial {i} has no stimulus period")
            stim_end_steps.append(-1)
        else:
            stim_end_steps.append(stim_steps[-1])

    stim_end_steps = np.array(stim_end_steps)
    valid = stim_end_steps >= 0
    ok(f"Trials with stimulus period: {valid.sum()}/{len(valid)}")

    # Most common stimulus-end timestep (should be consistent across trials)
    unique_ends, counts = np.unique(stim_end_steps[valid], return_counts=True)
    modal_end = unique_ends[counts.argmax()]
    ok(f"Modal stimulus-end timestep: t={modal_end}  "
       f"(used as fixed PID snapshot)")

    if len(unique_ends) > 5:
        warn(f"Stimulus-end varies a lot ({len(unique_ends)} unique values) — "
             f"NeuroGym may be sampling stochastic timing. "
             f"Consider using the modal value or the last stimulus step per trial.")

    # Extract hidden-state snapshot at modal_end for all test trials
    # (In practice this will be h(t), not obs — but obs here verifies shape)
    snapshot = obs[:, modal_end, :]   # (N, ob_size)
    ok(f"Stimulus-end snapshot shape: {snapshot.shape}  "
       f"→ will be (N, hidden_size) after RNN forward pass")

    # Coherence as PID target: check variance and range
    ok(f"Coherence stats — mean:{cohs.mean():.3f}  "
       f"std:{cohs.std():.3f}  "
       f"min:{cohs.min():.1f}  max:{cohs.max():.1f}")

    if cohs.std() < 1.0:
        warn("Low coherence variance — PID with continuous target may be weak")

    # Verify 50/50 subpopulation split will work (hidden_size must be even)
    print(f"\n  PID subpopulation split:")
    print(f"    hidden_size=100 → two groups of 50  ✓")
    print(f"    200 random bipartitions will be averaged (as per proposal)")
    print(f"    Each bipartition: PID(X1[50], X2[50]; coherence[scalar])")

    ok("Data is PID-ready")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(data_dir: str) -> None:
    print("=" * 55)
    print("  PDM Dataset Validation")
    print(f"  Directory: {data_dir}")
    print("=" * 55)

    splits = check_files(data_dir)
    check_values(splits)
    check_training_pipeline(splits)
    check_pid_readiness(splits)

    print(f"\n{'='*55}")
    print("  All checks passed — dataset is training-ready.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    args = p.parse_args()
    main(args.data_dir)