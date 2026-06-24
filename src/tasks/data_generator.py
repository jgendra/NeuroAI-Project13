"""
mante_generator.py
Generates the Full Context and Masked Perceptual datasets.
"""

import os
import json
import argparse
import numpy as np
import neurogym as ngym
import gymnasium as gym
from typing import Dict

from mante_config import CONFIG, UNIFORM_COHS, MANTE_TEST_COHS

def make_env(config: dict, seed: int, test_mode: str = "uniform") -> gym.Env:
    """
    Instantiates the NeuroGym environment with Mante et al. specifications.
    
    Args:
        config: Dictionary containing environment parameters.
        seed: Random seed for reproducibility.
        test_mode: "uniform" for dense coherences, "mante" for discrete coherences.
        
    Returns:
        Initialized NeuroGym environment.
    """
    cohs = MANTE_TEST_COHS if test_mode == "mante" else UNIFORM_COHS
    
    env = ngym.make(
        config["task"],
        dt=config["dt"],
        sigma=config["sigma"],
        timing=config["timing"],
        use_expl_context=True,
    )

    env.cohs = cohs

    env.reset(seed=seed)
    return env

# In data_generator.py

def extract_period_labels(env: gym.Env, seq_len: int) -> np.ndarray:
    """
    Extracts timestep-level period labels using exact mathematical timings.
    0 = Fixation, 1 = Stimulus, 2 = Decision
    """
    dt = env.unwrapped.dt
    timing = env.unwrapped.timing
    
    # Calculate duration boundaries in timesteps
    fix_steps = int(timing["fixation"] / dt)
    stim_steps = int(timing["stimulus"] / dt)
    
    arr = np.zeros(seq_len, dtype=np.int8)
    
    # 0 to fix_steps is implicitly 0 (Fixation)
    
    # Stimulus Period
    stim_start = fix_steps
    dec_start = fix_steps + stim_steps
    arr[stim_start:dec_start] = 1
    
    # Decision Period
    arr[dec_start:] = 2
    
    return arr

def generate_split(
    env: gym.Env, 
    n_trials: int, 
    seq_len: int, 
    is_perceptual: bool
) -> Dict[str, np.ndarray]:
    """
    Generates a dataset split. If is_perceptual is True, it masks the distractor stimulus.
    
    Args:
        env: The initialized NeuroGym environment.
        n_trials: Number of trials to generate.
        seq_len: Total timesteps per trial.
        is_perceptual: Boolean flag to apply the distractor mask.
        
    Returns:
        Dictionary of numpy arrays ready to be saved.
    """
    ob_size = env.observation_space.shape[0]
    
    observations = np.zeros((n_trials, seq_len, ob_size), dtype=np.float32)
    labels = np.zeros((n_trials, seq_len), dtype=np.int64)
    coherences = np.zeros((n_trials,), dtype=np.float32)
    contexts = np.zeros((n_trials,), dtype=np.int8)
    periods = np.zeros((n_trials, seq_len), dtype=np.int8)

    for i in range(n_trials):
        env.new_trial()
        
        ob = env.unwrapped.ob.copy()
        gt = env.unwrapped.gt.copy()
        trial_info = env.unwrapped.trial
        
        # In ContextDecisionMaking: Context 1 -> attend Modality 1, Context 2 -> attend Modality 2
        # (Neurogym internal contexts are usually 1 or -1, or 0/1 depending on the version. We map to 1 and 2)
        ctx = 1 if trial_info['context'] > 0 else 2
        
        if is_perceptual:
            # TRUE PERCEPTUAL MASKING: Eliminate all dynamic routing.
            # 1. Force the Context cue to ALWAYS be Context 1.
            # NeuroGym context channels are at indices 5 and 6.
            ob[:, 5] = 1.0  # Context 1 ON
            ob[:, 6] = 0.0  # Context 2 OFF
            
            # 2. If the original trial was a "Context 2" trial, 
            # move its stimulus data from channels 3/4 over to channels 1/2.
            ng_ctx = trial_info['context']
            if ng_ctx == 1:
                ob[:, 1:3] = ob[:, 3:5]
                
            # 3. Permanently zero out the distractor channels (3 and 4)
            ob[:, 3:5] = 0.0
            
            # Override the tracked context variable so the labels match
            ctx = 1

        T = min(ob.shape[0], seq_len)
        
        observations[i, :T, :] = ob[:T]
        labels[i, :T] = gt[:T]
        
        if T < seq_len:
            observations[i, T:, :] = ob[-1]
            labels[i, T:] = gt[-1]
            
        # Target Coherence (signed for Choice 1 / Choice 2)
        coh = float(trial_info.get("coh", 0.0))
        target_choice = trial_info.get("ground_truth", 1)
        coherences[i] = coh if target_choice == 1 else -coh
        
        contexts[i] = ctx
        periods[i] = extract_period_labels(env, seq_len)

        if (i + 1) % 5000 == 0:
            print(f"    Generated {i+1}/{n_trials} trials...")

    return {
        "observations": observations,
        "labels": labels,
        "coherences": coherences,
        "contexts": contexts,
        "trial_periods": periods,
    }

def main(args):
    # Update the master CONFIG with any CLI overrides
    CONFIG["dt"] = args.dt
    CONFIG["sigma"] = args.sigma
    CONFIG["splits"]["train"] = args.n_train
    CONFIG["splits"]["val"] = args.n_val
    CONFIG["splits"]["test_uniform"] = args.n_test_uni
    CONFIG["splits"]["test_mante"] = args.n_test_mante
    CONFIG["seed"] = args.seed

    # Allow saving to a custom directory (e.g., for tiny test runs)
    output_base = args.output_dir if args.output_dir else CONFIG["output_dir"]

    is_perceptual = (args.mode == "perceptual")
    out_dir = os.path.join(output_base, args.mode)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"=== Generating {args.mode.upper()} Dataset ===")
    
    rng = np.random.RandomState(CONFIG["seed"])
    
    for split_name, n_trials in CONFIG["splits"].items():
        if n_trials <= 0:
            continue # Skip generation if user sets trials to 0

        print(f"\nProcessing Split: {split_name} ({n_trials} trials)")
        
        seed = int(rng.randint(0, 2**31))
        test_mode = "mante" if "mante" in split_name else "uniform"
        env = make_env(CONFIG, seed, test_mode)
        
        data = generate_split(env, n_trials, CONFIG["seq_len"], is_perceptual)
        
        path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez_compressed(path, **data)
        print(f"  Saved to {path} ({(os.path.getsize(path)/1e6):.1f} MB)")

    # Save the final configuration as a readable JSON file
    config_save_path = os.path.join(out_dir, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(CONFIG, f, indent=4)
    print(f"\nConfiguration saved to {config_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["context", "perceptual"], required=True)

    # CLI Overrides
    parser.add_argument("--n_train", type=int, default=CONFIG["splits"]["train"], help="Number of training trials")
    parser.add_argument("--n_val", type=int, default=CONFIG["splits"]["val"], help="Number of validation trials")
    parser.add_argument("--n_test_uni", type=int, default=CONFIG["splits"]["test_uniform"], help="Number of uniform test trials")
    parser.add_argument("--n_test_mante", type=int, default=CONFIG["splits"]["test_mante"], help="Number of Mante discrete test trials")
    parser.add_argument("--dt", type=int, default=CONFIG["dt"], help="Integration timestep in ms")
    parser.add_argument("--sigma", type=float, default=CONFIG["sigma"], help="Noise standard deviation")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")

    args = parser.parse_args()
    main(args)
