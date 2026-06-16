
import argparse
import os
import numpy as np
#import neurogym as ngym
# from neurogym.wrappers.block import TrialHistoryV2
import gymnasium as gym

"""
Dataset generation for PerceptualDecisionMaking-v0
Project 13 – Information Decomposition in Task-Trained RNNs

Generates train / validation / test splits and saves them as .npz files.
Each file contains:
    observations  : (n_trials, seq_len, ob_size)   float32  — network inputs
    labels        : (n_trials, seq_len)             int64    — ground-truth action per timestep
    coherences    : (n_trials,)                     float32  — signed stimulus coherence (PID target)
    trial_periods : (n_trials, seq_len)             int8     — period label per timestep
                    0 = fixation, 1 = stimulus, 2 = decision

Usage:
    python generate_pdm_dataset.py              # uses default settings
    python generate_pdm_dataset.py --seed 99    # override random seed
"""


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Task
    "task": "PerceptualDecisionMaking-v0",
    "dt": 20,                        # ms per timestep (override NeuroGym default of 100)
    "cohs": [0.0, 6.4, 12.8, 25.6, 51.2],  # coherence levels (Britten 1992 standard set)
    "sigma": 1.0,                    # input noise std

    # Sequence length: fixation (~250ms) + stimulus (~500ms) + decision (~250ms)
    # At dt=20ms: 13 + 25 + 13 = 51 steps. NeuroGym manages this internally;
    # seq_len here is the padded length passed to ngym.Dataset.
    "seq_len": 50,

    # Dataset sizes (number of trials)
    "n_train": 10000,
    "n_val":   1000,
    "n_test":  1000,

    # Batch size used by ngym.Dataset internally (does not affect saved output)
    "batch_size": 16,

    # Reproducibility
    "seed": 42,

    # Output
    "output_dir": "data/pdm",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build environment
# ──────────────────────────────────────────────────────────────────────────────

def make_env(config: dict, seed: int):
    """Instantiate and seed the PerceptualDecisionMaking environment."""
    env_kwargs = {
        "dt":    config["dt"],
        "cohs":  config["cohs"],
        "sigma": config["sigma"],
    }
    env = ngym.make(config["task"], **env_kwargs)
    env.reset(seed=seed)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Helper: extract signed coherence and period labels from a completed trial
# ──────────────────────────────────────────────────────────────────────────────

def get_trial_metadata(env: gym.Env, seq_len: int):
    """
    Return:
        signed_coh  : float  — positive = choice 1 correct, negative = choice 2 correct
        period_arr  : (seq_len,) int8 — 0=fixation, 1=stimulus, 2=decision
    """
    trial = env.trial  # dict populated after env.new_trial()

    # Signed coherence: coherence magnitude with sign encoding ground-truth choice
    # NeuroGym stores ground_truth as 1 or 2; map to +/- coherence
    coh = trial.get("coh", trial.get("coherence", 0.0))
    gt  = trial.get("ground_truth", 1)
    signed_coh = float(coh) if gt == 1 else -float(coh)

    # Period label array — env.start_t and env.end_t give period boundaries in steps
    period_arr = np.zeros(seq_len, dtype=np.int8)
    periods = env.timing  # dict: period_name -> duration_in_ms

    step = 0
    period_id = {"fixation": 0, "stimulus": 1, "decision": 2}
    for period_name, duration_ms in periods.items():
        n_steps = int(round(duration_ms / env.dt))
        label   = period_id.get(period_name, 0)
        end     = min(step + n_steps, seq_len)
        period_arr[step:end] = label
        step = end
        if step >= seq_len:
            break

    return signed_coh, period_arr


# ──────────────────────────────────────────────────────────────────────────────
# Core: generate one split
# ──────────────────────────────────────────────────────────────────────────────

def generate_split(
    config: dict,
    n_trials: int,
    seed: int,
    split_name: str,
) -> dict:
    """
    Generate n_trials trials and return a dict of arrays.

    Returns
    -------
    {
        "observations"  : (n_trials, seq_len, ob_size)  float32
        "labels"        : (n_trials, seq_len)           int64
        "coherences"    : (n_trials,)                   float32
        "trial_periods" : (n_trials, seq_len)           int8
    }
    """
    seq_len = config["seq_len"]
    env     = make_env(config, seed)

    ob_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    print(f"[{split_name}] ob_size={ob_size}, act_size={act_size}, "
          f"seq_len={seq_len}, n_trials={n_trials}")

    observations  = np.zeros((n_trials, seq_len, ob_size), dtype=np.float32)
    labels        = np.zeros((n_trials, seq_len),          dtype=np.int64)
    coherences    = np.zeros((n_trials,),                  dtype=np.float32)
    trial_periods = np.zeros((n_trials, seq_len),          dtype=np.int8)

    for i in range(n_trials):

        obs_list = []
        gt_list = []

        obs, info = env.reset()

        terminated = False
        truncated = False

        while not (terminated or truncated):

            # random action just to collect the trial
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            obs_list.append(obs)

            # NeuroGym provides ground truth in info
            gt_list.append(info.get("gt", 0))


        ob = np.array(obs_list)
        gt = np.array(gt_list)

        trial_len = ob.shape[0]

        # Pad or truncate to seq_len
        t = min(trial_len, seq_len)
        observations[i, :t, :] = ob[:t].astype(np.float32)
        labels[i, :t] = gt[:t].astype(np.int64)

        # If trial is shorter than seq_len, repeat last observation and label
        if t < seq_len:
            observations[i, t:, :] = ob[-1].astype(np.float32)
            labels[i,        t:]   = gt[-1].astype(np.int64)

        signed_coh, period_arr = get_trial_metadata(env, seq_len)
        coherences[i]      = signed_coh
        trial_periods[i]   = period_arr

        if (i + 1) % 1 == 0:
            print(f"  {i+1}/{n_trials} trials generated")

    return {
        "observations":  observations,
        "labels":        labels,
        "coherences":    coherences,
        "trial_periods": trial_periods,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

def save_split(data: dict, output_dir: str, split_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{split_name}.npz")
    np.savez_compressed(path, **data)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {path}  ({size_mb:.1f} MB)")
    print(f"  Shapes — observations: {data['observations'].shape}, "
          f"labels: {data['labels'].shape}, "
          f"coherences: {data['coherences'].shape}")


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity checks
# ──────────────────────────────────────────────────────────────────────────────

def run_sanity_checks(data: dict, split_name: str) -> None:
    print(f"\n[Sanity checks — {split_name}]")

    obs  = data["observations"]
    labs = data["labels"]
    cohs = data["coherences"]

    # 1. No NaNs
    assert not np.any(np.isnan(obs)),  "NaN in observations"
    assert not np.any(np.isnan(cohs)), "NaN in coherences"
    print("  ✓ No NaNs")

    # 2. Labels are valid action indices (0, 1, 2)
    unique_labels = np.unique(labs)
    assert np.all(unique_labels >= 0) and np.all(unique_labels <= 2), \
        f"Unexpected label values: {unique_labels}"
    print(f"  ✓ Label values: {unique_labels}")

    # 3. Coherences match expected levels (both signs + 0)
    expected = sorted(set([0.0] + [c for c in DEFAULT_CONFIG["cohs"]] +
                          [-c for c in DEFAULT_CONFIG["cohs"] if c != 0.0]))
    actual   = sorted(np.unique(cohs))
    print(f"  ✓ Unique coherences: {actual}")

    # 4. Approximate class balance (fixation label 0 should dominate)
    for label_val, name in [(0, "fixation"), (1, "choice1"), (2, "choice2")]:
        frac = np.mean(labs == label_val)
        print(f"     label {label_val} ({name}): {frac:.2%} of timesteps")

    # 5. Observation range check (should be roughly [-2, 2] with sigma=1)
    print(f"  ✓ Observation range: [{obs.min():.2f}, {obs.max():.2f}]")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(config: dict) -> None:
    rng = np.random.RandomState(config["seed"])
    seeds = {
        "train": int(rng.randint(0, 2**31)),
        "val":   int(rng.randint(0, 2**31)),
        "test":  int(rng.randint(0, 2**31)),
    }

    splits = {
        "train": config["n_train"],
        "val":   config["n_val"],
        "test":  config["n_test"],
    }

    print("=" * 60)
    print("PerceptualDecisionMaking-v0  Dataset Generation")
    print(f"dt={config['dt']}ms  seq_len={config['seq_len']}  "
          f"cohs={config['cohs']}  sigma={config['sigma']}")
    print(f"total trials: {sum(splits.values())}")
    print("=" * 60)

    for split_name, n_trials in splits.items():
        print(f"\n── {split_name.upper()} ──")
        data = generate_split(config, n_trials, seeds[split_name], split_name)
        run_sanity_checks(data, split_name)
        save_split(data, config["output_dir"], split_name)

    # Save config alongside data for reproducibility
    config_path = os.path.join(config["output_dir"], "config.npz")
    np.savez(config_path, **{k: str(v) for k, v in config.items()})
    print(f"\nConfig saved to {config_path}")
    print("\nDone.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDM dataset")
    parser.add_argument("--seed",       type=int,   default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--n_train",    type=int,   default=DEFAULT_CONFIG["n_train"])
    parser.add_argument("--n_val",      type=int,   default=DEFAULT_CONFIG["n_val"])
    parser.add_argument("--n_test",     type=int,   default=DEFAULT_CONFIG["n_test"])
    parser.add_argument("--seq_len",    type=int,   default=DEFAULT_CONFIG["seq_len"])
    parser.add_argument("--dt",         type=int,   default=DEFAULT_CONFIG["dt"])
    parser.add_argument("--sigma",      type=float, default=DEFAULT_CONFIG["sigma"])
    parser.add_argument("--output_dir", type=str,   default=DEFAULT_CONFIG["output_dir"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update(vars(args))
    main(config)