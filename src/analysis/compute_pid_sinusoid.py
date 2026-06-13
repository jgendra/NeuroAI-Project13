"""
compute_pid_sinusoid.py
=======================

Test the Gaussian analytic PID implementation on the sinusoid-trained RNN.

Goal: compute PID at the *last timestep* of the RNN's hidden-state trajectory,
using the hidden activations as the two source groups and the input sinusoid as
the target.

Important modelling note
------------------------
PID is estimated from a *covariance over an ensemble of observations*. A single
timestep of a single trajectory is one observation, which cannot define a
covariance. The sinusoid RNN was trained on one trajectory, so to obtain a valid
"PID at the last timestep" we must build an ensemble. We do this exactly the way
the real NeuroGym project will: we run the trained RNN over many *trials* and use
the trial dimension as the sample/ensemble axis. Each trial is the same sinusoid
with a random phase and a small amount of input noise (mirroring the noisy,
parametrically-varied stimuli of the decision-making tasks). The PID at the last
timestep is then computed across trials:

    sources = hidden activations h(T) at the last timestep   (n_trials x n_units)
    target  = the (clean) sinusoid value at the last timestep (n_trials,)

Run `main.py` first so that `RNN_Sinusoid.pt` exists.
"""

import torch
import numpy as np
from trueRNN import TrueRNN
from gaussian_pid import gaussian_pid_rnn

# --------------------------------------------------------------------------- #
# Settings (kept explicit so every choice is visible)
# --------------------------------------------------------------------------- #
SYSTEM = "Sinusoid"          # which trained model file to load (RNN_<SYSTEM>.pt)
DIM = 1                       # the sinusoid is a single (1-D) channel
N_TIMESTEPS = 300             # trajectory length, must match how the RNN was trained
PERIOD = 50.0                 # sinusoid period in timesteps, must match training
N_TRIALS = 500                # ensemble size: number of phase-randomised trials
INPUT_NOISE_STD = 0.05        # std of additive input noise per timestep (keeps MI finite)
N_BIPARTITIONS = 200          # number of random 5/5 unit splits to average PID over
SEED = 0                      # reproducibility for phases, noise and bipartitions
LOG_BASE = 2                  # report information in bits

if __name__ == "__main__":

    # --- reproducibility ---------------------------------------------------- #
    rng = np.random.default_rng(SEED)          # numpy RNG for phases/noise
    torch.manual_seed(SEED)                    # torch RNG (defensive; eval is deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU is fine

    # --- load the trained sinusoid RNN -------------------------------------- #
    # The constructor's default hidden size is 10, matching the trained model.
    model = TrueRNN(dim=DIM, system=SYSTEM).to(device)         # build the architecture
    model_path = f"RNN_{SYSTEM}.pt"                            # path to the saved weights
    state = torch.load(model_path, map_location=device, weights_only=True)  # load weights
    model.load_state_dict(state)                               # restore the parameters
    model.eval()                                               # inference mode
    n_units = model.hidden_dim                                 # number of hidden units (10)

    # --- build the trial ensemble (phase-randomised, slightly noisy sinusoids) #
    t = np.arange(N_TIMESTEPS)                                 # timestep index 0..T-1
    phases = rng.uniform(0.0, PERIOD, size=N_TRIALS)           # one random phase per trial
    # clean target signal per trial and timestep: sin(2*pi*(t + phase)/period)
    clean = np.sin(2 * np.pi * (t[None, :] + phases[:, None]) / PERIOD)   # (n_trials, T)
    # the RNN input is the clean signal plus small observation noise
    noise = INPUT_NOISE_STD * rng.standard_normal((N_TRIALS, N_TIMESTEPS)) # (n_trials, T)
    inputs_np = clean + noise                                  # noisy input stream (n_trials, T)

    # --- run the RNN over the whole ensemble in one batched pass ------------ #
    # nn.RNN (batch_first=False) expects input shaped (seq_len, batch, input_size).
    x = torch.tensor(inputs_np.T[:, :, None], dtype=torch.float32, device=device)  # (T, n_trials, 1)
    with torch.no_grad():                                      # no gradients needed at eval
        h_seq, _ = model.rnn(x)                                # hidden states: (T, n_trials, n_units)
    # reorder to the wrapper's convention (n_samples, n_timesteps, n_units)
    activations = h_seq.permute(1, 0, 2).cpu().numpy()         # (n_trials, T, n_units)

    # --- target: the clean sinusoid value (the underlying signal, like coherence) #
    target = clean                                             # (n_trials, T); univariate per step

    # --- compute PID at the LAST timestep, averaged over random 5/5 splits --- #
    out = gaussian_pid_rnn(
        activations,                 # sources: (n_trials, T, n_units)
        target,                      # target : (n_trials, T)
        timestep=-1,                 # -1 selects the last timestep
        bipartitions="random",       # average over random balanced unit splits
        n_bipartitions=N_BIPARTITIONS,
        balanced=True,               # 5 units vs 5 units for a 10-unit RNN
        seed=SEED,
        log_base=LOG_BASE,           # bits
    )

    # --- report ------------------------------------------------------------- #
    unit = "bits" if LOG_BASE == 2 else "nats"
    print(f"Gaussian analytic PID at the last timestep (t = {N_TIMESTEPS - 1})")
    print(f"  ensemble: {N_TRIALS} phase-randomised trials, "
          f"{n_units} units split 5/5 over {N_BIPARTITIONS} bipartitions")
    print(f"  (values in {unit}; '+/-' is the spread across bipartitions)\n")
    print(f"  Redundancy : {out['redundancy']:.4f} +/- {out['redundancy_std']:.4f}")
    print(f"  Unique 1   : {out['unique1']:.4f} +/- {out['unique1_std']:.4f}")
    print(f"  Unique 2   : {out['unique2']:.4f} +/- {out['unique2_std']:.4f}")
    print(f"  Synergy    : {out['synergy']:.4f} +/- {out['synergy_std']:.4f}")
    print(f"  -----")
    print(f"  I(X1;Y)    : {out['mi_1']:.4f}")
    print(f"  I(X2;Y)    : {out['mi_2']:.4f}")
    print(f"  I(X1,X2;Y) : {out['mi_joint']:.4f}  (= sum of the four atoms)")
