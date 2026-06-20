"""
Dataset generation for PerceptualDecisionMaking-v0
Project 13 – Information Decomposition in Task-Trained RNNs
NeuroGym 2.3.1 / gymnasium API

Correct approach: use env.new_trial() directly to get env.ob and env.gt
per trial, rather than stepping through with env.step() (which is for RL
and gives unreliable gt labels in info).

Saved arrays per split:
    observations  : (n_trials, seq_len, ob_size)  float32
    labels        : (n_trials, seq_len)            int64    — 0=fixate, 1=choice1, 2=choice2
    coherences    : (n_trials,)                    float32  — signed (PID target variable)
    trial_periods : (n_trials, seq_len)            int8     — 0=fixation, 1=stimulus, 2=decision
"""

import argparse
import os
import numpy as np
import neurogym as ngym

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "task":       "PerceptualDecisionMaking-v0",
    "dt":         20,                              # ms; overrides NeuroGym default of 100
    "cohs":       [0.0, 6.4, 12.8, 25.6, 51.2],  # Britten 1992 standard levels
    "sigma":      1.0,
    "seq_len":    120,   # timesteps @ dt=20ms → 2000ms total per trial (safe upper bound)
    "n_train":    10000,
    "n_val":      1000,
    "n_test":     1000,
    "seed":       42,
    "output_dir": "data/pdm",
    "print_every": 500,
}


# ──────────────────────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────────────────────

def make_env(config: dict, seed: int):
    env = ngym.make(
        config["task"],
        dt=config["dt"],
        cohs=config["cohs"],
        sigma=config["sigma"],
    )
    env.reset(seed=seed)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Period label array from env.timing
# ──────────────────────────────────────────────────────────────────────────────

def get_period_array(env, seq_len: int) -> np.ndarray:
    """Infer period labels directly from env.gt — no timing key assumptions."""
    gt  = env.unwrapped.gt
    ob  = env.unwrapped.ob
    T   = len(gt)
    arr = np.zeros(seq_len, dtype=np.int8)
    t   = min(T, seq_len)

    # fixation = where ch0 (fixation channel) is 1
    fix = ob[:t, 0] > 0.5
    arr[:t][fix] = 0

    # stimulus = ch0 is 0 AND gt is still 0
    stim = (~fix) & (gt[:t] == 0)
    arr[:t][stim] = 1

    # decision = where gt becomes non-zero
    dec = gt[:t] != 0
    arr[:t][dec] = 2

    # pad remainder with last value
    if t < seq_len:
        arr[t:] = arr[t - 1]

    return arr

# ──────────────────────────────────────────────────────────────────────────────
# Signed coherence from trial dict
# ──────────────────────────────────────────────────────────────────────────────

def signed_coherence(trial: dict) -> float:
    """
    NeuroGym stores the coherence magnitude and the ground-truth choice (1 or 2).
    We encode direction into the sign: positive = choice 1, negative = choice 2.
    This is the continuous PID target variable.
    """
    coh = float(trial.get("coh", trial.get("coherence", 0.0)))
    gt  = trial.get("ground_truth", 1)
    return coh if gt == 1 else -coh


# ──────────────────────────────────────────────────────────────────────────────
# Core generation loop
# ──────────────────────────────────────────────────────────────────────────────

def generate_split(config: dict, n_trials: int, seed: int, name: str) -> dict:
    seq_len = config["seq_len"]
    env     = make_env(config, seed)

    ob_size = env.observation_space.shape[0]
    print(f"[{name}] ob_size={ob_size}  act_size={env.action_space.n}  "
          f"seq_len={seq_len}  n_trials={n_trials}")

    observations  = np.zeros((n_trials, seq_len, ob_size), dtype=np.float32)
    labels        = np.zeros((n_trials, seq_len),          dtype=np.int64)
    coherences    = np.zeros((n_trials,),                  dtype=np.float32)
    trial_periods = np.zeros((n_trials, seq_len),          dtype=np.int8)

    for i in range(n_trials):
        # ── KEY FIX: use new_trial() directly ──────────────────────────────
        # env.ob  : (trial_timesteps, ob_size)  — full observation sequence
        # env.gt  : (trial_timesteps,)          — ground-truth action per step
        # This is the correct supervised-learning API in NeuroGym.
        # Do NOT use env.step() — it's for RL and gives unreliable gt in info.
        env.new_trial()


        ob = env.ob   # (T, ob_size)
        gt = env.gt   # (T,)  values: 0=fixate, 1=choice1, 2=choice2
        # with:
        env.unwrapped.new_trial()
        ob = env.unwrapped.ob
        gt = env.unwrapped.gt.copy()
        
        T = ob.shape[0]
        t = min(T, seq_len)

        observations[i, :t, :]  = ob[:t].astype(np.float32)
        labels[i,        :t]    = gt[:t].astype(np.int64)

        # Pad to seq_len by repeating last timestep
        if t < seq_len:
            observations[i, t:, :] = ob[-1].astype(np.float32)
            labels[i,        t:]   = gt[-1].astype(np.int64)

        coherences[i]      = signed_coherence(env.trial)
        trial_periods[i]   = get_period_array(env, seq_len)

        if (i + 1) % config["print_every"] == 0 or (i + 1) == n_trials:
            print(f"  {i+1:>{len(str(n_trials))}}/{n_trials}")

    return {
        "observations":  observations,
        "labels":        labels,
        "coherences":    coherences,
        "trial_periods": trial_periods,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

def save_split(data: dict, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.npz")
    np.savez_compressed(path, **data)
    mb = os.path.getsize(path) / 1e6
    print(f"  → {path}  ({mb:.1f} MB)")
    for k, v in data.items():
        print(f"     {k}: {v.shape}  {v.dtype}")


# ──────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────────────────────────────────────

def sanity_check(data: dict, name: str, config: dict) -> None:
    print(f"\n[Sanity — {name}]")
    obs, labs, cohs, periods = (
        data["observations"], data["labels"],
        data["coherences"],   data["trial_periods"],
    )

    # NaNs
    assert not np.isnan(obs).any(),  "NaN in observations"
    assert not np.isnan(cohs).any(), "NaN in coherences"
    print("  ✓ No NaNs")

    # Label range
    unique_labs = np.unique(labs)
    assert unique_labs.min() >= 0 and unique_labs.max() <= 2, \
        f"Bad labels: {unique_labs}"
    print(f"  ✓ Label values: {unique_labs}")

    # Coherence values (both signs + 0 for coh=0)
    print(f"  ✓ Coherences: {sorted(np.unique(cohs))}")

    # Period coverage
    unique_periods = np.unique(periods)
    print(f"  ✓ Periods present: {unique_periods}  (0=fix, 1=stim, 2=dec)")

    # Per-label timestep fractions
    for lv, ln in [(0, "fixate"), (1, "choice1"), (2, "choice2")]:
        print(f"     label {lv} ({ln}): {np.mean(labs == lv):.1%} of timesteps")

    # Observation range
    print(f"  ✓ Obs range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Trial length check: fixation period should always start at t=0
    assert np.all(periods[:, 0] == 0), "t=0 is not fixation period"
    print("  ✓ All trials start in fixation")

    # Decision period exists in every trial
    has_decision = np.any(periods == 2, axis=1)
    if not has_decision.all():
        n_missing = (~has_decision).sum()
        print(f"  ⚠ {n_missing} trials have no decision period "
              f"(seq_len={config['seq_len']} may be too short — consider increasing it)")
    else:
        print("  ✓ Decision period present in all trials")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(config: dict) -> None:
    rng = np.random.RandomState(config["seed"])
    seeds = {s: int(rng.randint(0, 2**31))
             for s in ("train", "val", "test")}

    splits = {"train": config["n_train"],
              "val":   config["n_val"],
              "test":  config["n_test"]}

    print("=" * 60)
    print("PerceptualDecisionMaking-v0  –  Dataset Generation")
    print(f"dt={config['dt']}ms | seq_len={config['seq_len']} | "
          f"cohs={config['cohs']} | sigma={config['sigma']}")
    print(f"total trials: {sum(splits.values())}")
    print("=" * 60)

    for split_name, n_trials in splits.items():
        print(f"\n── {split_name.upper()} ──")
        data = generate_split(config, n_trials, seeds[split_name], split_name)
        sanity_check(data, split_name, config)
        save_split(data, config["output_dir"], split_name)

    # Save config for reproducibility
    cfg_path = os.path.join(config["output_dir"], "config.npz")
    np.savez(cfg_path, **{k: str(v) for k, v in config.items()})
    print(f"\nConfig → {cfg_path}")
    print("Done.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",        type=int,   default=DEFAULT_CONFIG["seed"])
    p.add_argument("--n_train",     type=int,   default=DEFAULT_CONFIG["n_train"])
    p.add_argument("--n_val",       type=int,   default=DEFAULT_CONFIG["n_val"])
    p.add_argument("--n_test",      type=int,   default=DEFAULT_CONFIG["n_test"])
    p.add_argument("--seq_len",     type=int,   default=DEFAULT_CONFIG["seq_len"])
    p.add_argument("--dt",          type=int,   default=DEFAULT_CONFIG["dt"])
    p.add_argument("--sigma",       type=float, default=DEFAULT_CONFIG["sigma"])
    p.add_argument("--output_dir",  type=str,   default=DEFAULT_CONFIG["output_dir"])
    p.add_argument("--print_every", type=int,   default=DEFAULT_CONFIG["print_every"])
    args = p.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(vars(args))
    main(cfg)