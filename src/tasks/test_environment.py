import numpy as np
import neurogym as ngym

def test_environment():
    print("=== NeuroGym Environment Verification Test ===\n")
    
    # 1. Setup the environment exactly like your config
    config = {
        "dt": 10,
        "sigma": 1.0, # High noise regime
        "timing": {"fixation": 300, "stimulus": 750, "delay": 0, "decision": 100}
    }
    
    # We will test with the classic Mante values
    test_cohs = [5, 15, 50] 
    
    env = ngym.make(
        'ContextDecisionMaking-v0',
        dt=config["dt"],
        sigma=config["sigma"],
        timing=config["timing"],
        use_expl_context=True
    )
    env.unwrapped.cohs = test_cohs
    
    # Calculate stimulus indices based on dt
    stim_start_idx = int(config["timing"]["fixation"] / config["dt"])
    stim_end_idx = stim_start_idx + int(config["timing"]["stimulus"] / config["dt"])

    # =====================================================================
    # TEST 1: Coherence Mean vs. Noisy Stimulus Mean
    # =====================================================================
    print("--- TEST 1: Signal vs. Noise Recovery ---")
    print("If coh=15, the theoretical mean difference between channels is 0.15\n")
    
    env.seed(42)
    env.reset()
    
    # We'll run 5 trials to see how well the mean recovers the signal
    for i in range(5):
        env.new_trial()
        ob = env.unwrapped.ob
        trial_info = env.unwrapped.trial
        
        ground_truth = trial_info["ground_truth"] # 1 (Choice A) or 2 (Choice B)

        if trial_info['context'] == 0:
            stim_period = ob[stim_start_idx:stim_end_idx, 1:3]
            coh = float(trial_info["coh_1"])
            mod_name = "Modality 1"
        else:
            stim_period = ob[stim_start_idx:stim_end_idx, 3:5]
            coh = float(trial_info["coh_2"])
            mod_name = "Modality 2"
        
        # Channel 1 is Choice A, Channel 2 is Choice B
        chan1_mean = np.mean(stim_period[:, 0])
        chan2_mean = np.mean(stim_period[:, 1])
        
        # Calculate the empirical difference 
        # (Positive means evidence points to Choice A, Negative points to Choice B)
        empirical_diff = chan1_mean - chan2_mean
        
        # Theoretical difference based on NeuroGym's internal math
        theoretical_diff = (coh / 100.0) if ground_truth == 1 else -(coh / 100.0)
        
        print(f"Trial {i+1}| Context: {mod_name} | Target: Choice {ground_truth} | Raw coh: {coh}")
        print(f"  -> Expected Mean Diff: {theoretical_diff: .4f}")
        print(f"  -> Actual Mean Diff:   {empirical_diff: .4f}  (Error: {abs(theoretical_diff - empirical_diff):.4f})")
    
    
    # =====================================================================
    # TEST 2: Determinism (Seed Stability)
    # =====================================================================
    print("\n\n--- TEST 2: RNG Determinism ---")
    print("Generating a sequence of 10 trials, resetting the seed, and generating again...")
    
    def generate_sequence(seed, n_trials=10):
        env.seed(seed)
        env.reset()
        sequence_obs = []
        for _ in range(n_trials):
            env.new_trial()
            sequence_obs.append(env.unwrapped.ob.copy())
        return np.array(sequence_obs)

    # Generate Sequence A
    seq_A = generate_sequence(seed=99)
    
    # Generate Sequence B with the SAME seed
    seq_B = generate_sequence(seed=99)
    
    # Generate Sequence C with a DIFFERENT seed
    seq_C = generate_sequence(seed=100)

    # Assertions
    is_A_equals_B = np.array_equal(seq_A, seq_B)
    is_A_equals_C = np.array_equal(seq_A, seq_C)
    
    print(f"Sequence A == Sequence B (Same Seed):      {'PASS' if is_A_equals_B else 'FAIL'}")
    print(f"Sequence A == Sequence C (Different Seed): {'PASS' if not is_A_equals_C else 'FAIL'}")

if __name__ == "__main__":
    test_environment()
