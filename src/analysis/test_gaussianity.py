"""
test_gaussianity.py
===================

Tests whether the hidden-unit activations of the sinusoid-trained RNN are
approximately Gaussian distributed across trials at the final timestep.

Why this is the right check
---------------------------
Gaussian PID assumes that the joint distribution of (X1, X2, Y) is multivariate
Gaussian across the *trial* ensemble.  The relevant random variable for each unit
is therefore h_i(t*) evaluated across many independent trials (each with a
different phase and noise draw), NOT the activation of unit i across timesteps
within a single trajectory.  The latter is autocorrelated and mixes temporal
dynamics with the distributional question; the former is the actual sample the
covariance matrix is built from.

What this script does
---------------------
1.  Rebuilds the same 500-trial ensemble used in compute_pid_sinusoid.py.
2.  Extracts the across-trial snapshot at the final timestep: shape (500, 10).
3.  For each unit:
      - plots a histogram of its 500 across-trial activations
      - overlays the best-fit Gaussian
      - runs a Shapiro-Wilk normality test  (exact for n<=5000)
      - runs a D'Agostino-Pearson K² test   (sensitive to skew + kurtosis)
      - prints both p-values; p > 0.05 is consistent with Gaussianity
4.  Produces a Q-Q plot for each unit (points on the diagonal = Gaussian).
5.  Saves both figures.

Run `main.py` first so that `RNN_<SYSTEM>.pt` exists.
"""

import time
start_time = time.time()

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from RNN import ElmanRNN

# Define directories
dir = "src/analysis/results/"

# --------------------------------------------------------------------------- #
# Settings  — must match compute_pid_sinusoid.py exactly so the ensemble is
# identical and the gaussianity check applies to the same data PID is run on
# --------------------------------------------------------------------------- #
SYSTEM         = "Sinusoid"
DIM            = 1
N_TIMESTEPS    = 300
PERIOD         = 50.0
N_TRIALS       = 500
INPUT_NOISE_STD = 0.05
SEED           = 0

if __name__ == "__main__":

    # --- reproducibility ---------------------------------------------------- #
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load the trained RNN ----------------------------------------------- #
    model = ElmanRNN(dim=DIM, system=SYSTEM).to(device)
    model_path = f"{dir}RNN_{SYSTEM}.pt"
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_units = model.hidden_dim      # 10

    # --- build the identical trial ensemble --------------------------------- #
    # Must use the same SEED as compute_pid_sinusoid.py so this is checking the
    # exact distribution PID was estimated from.
    t_idx  = np.arange(N_TIMESTEPS)
    phases = rng.uniform(0.0, PERIOD, size=N_TRIALS)
    clean  = np.sin(2 * np.pi * (t_idx[None, :] + phases[:, None]) / PERIOD)
    noise  = INPUT_NOISE_STD * rng.standard_normal((N_TRIALS, N_TIMESTEPS))
    inputs_np = clean + noise                                  # (n_trials, T)

    # --- run the RNN over all trials in one batched pass ------------------- #
    x = torch.tensor(inputs_np.T[:, :, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        h_seq, _ = model.rnn(x)                                # (T, n_trials, n_units)
    activations = h_seq.permute(1, 0, 2).cpu().numpy()         # (n_trials, T, n_units)

    # --- extract the snapshot at the FINAL timestep ------------------------- #
    # This is the (500, 10) matrix whose covariance Gaussian PID is built from.
    # Each column is one unit's activation across 500 independent trials at t*.
    snapshot = activations[:, -1, :]                           # (n_trials, n_units)

    # --- normality tests per unit ------------------------------------------- #
    print(f"Normality tests on across-trial activations at t = {N_TIMESTEPS - 1}")
    print(f"(n = {N_TRIALS} trials per unit)\n")
    print(f"{'Unit':<6} {'SW p-val':>10} {'SW pass':>8}   {'DP p-val':>10} {'DP pass':>8}")
    print("-" * 52)

    sw_pvals = []
    dp_pvals = []
    for i in range(n_units):
        unit_acts = snapshot[:, i]                             # 500 across-trial values

        # Shapiro-Wilk: general normality test, exact for n<=5000
        _, sw_p = stats.shapiro(unit_acts)

        # D'Agostino-Pearson K²: tests skewness and excess kurtosis jointly;
        # complementary to Shapiro-Wilk which is more sensitive to tail behaviour
        _, dp_p = stats.normaltest(unit_acts)

        sw_pvals.append(sw_p)
        dp_pvals.append(dp_p)

        sw_pass = "YES" if sw_p > 0.05 else "NO "
        dp_pass = "YES" if dp_p > 0.05 else "NO "
        print(f"  {i+1:<4} {sw_p:>10.4f} {sw_pass:>8}   {dp_p:>10.4f} {dp_pass:>8}")

    n_pass_sw = sum(p > 0.05 for p in sw_pvals)
    n_pass_dp = sum(p > 0.05 for p in dp_pvals)
    print(f"\n  {n_pass_sw}/{n_units} units pass Shapiro-Wilk at α=0.05")
    print(f"  {n_pass_dp}/{n_units} units pass D'Agostino-Pearson at α=0.05")
    print(f"\n  Interpretation: p > 0.05 means we cannot reject Gaussianity.")
    print(f"  Units that fail both tests are the most problematic for Gaussian PID.")

    # ======================================================================== #
    # FIGURE 1: histograms + fitted Gaussian overlay, one panel per unit
    # ======================================================================== #
    fig1, axes = plt.subplots(2, 5, figsize=(14, 5.5))
    axes = axes.ravel()

    for i in range(n_units):
        ax   = axes[i]
        vals = snapshot[:, i]
        mu, sigma = vals.mean(), vals.std()

        # histogram of across-trial activations
        ax.hist(vals, bins=25, color='#4C78A8', alpha=0.75,
                density=True, zorder=2)

        # best-fit Gaussian overlay
        x_fit = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(x_fit, stats.norm.pdf(x_fit, mu, sigma),
                color='#E45756', lw=1.8, zorder=3)

        # Shapiro-Wilk result in title; red title = non-Gaussian
        sw_p  = sw_pvals[i]
        color = '#333333' if sw_p > 0.05 else '#E45756'
        ax.set_title(f'Unit {i+1}   SW p={sw_p:.3f}',
                     fontsize=9, color=color)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('activation', fontsize=7)

    fig1.suptitle(
        f'Across-trial activation distributions at t* = {N_TIMESTEPS-1}  '
        f'(N={N_TRIALS} trials)\n'
        'Blue = data histogram   Red = fitted Gaussian   '
        'Red title = Shapiro-Wilk rejects Gaussianity (p < 0.05)',
        fontsize=10
    )
    fig1.tight_layout()
    # fig1.savefig(f'{dir}gaussianity_histograms_{SYSTEM}.png',
    #              dpi=300, bbox_inches='tight')

    # ======================================================================== #
    # FIGURE 2: Q-Q plots, one panel per unit
    # A Q-Q plot maps the empirical quantiles against the theoretical Gaussian
    # quantiles. Points lying on the diagonal line indicate Gaussianity.
    # Deviations at the tails indicate heavier/lighter tails than Gaussian.
    # ======================================================================== #
    fig2, axes2 = plt.subplots(2, 5, figsize=(14, 5.5))
    axes2 = axes2.ravel()

    for i in range(n_units):
        ax   = axes2[i]
        vals = snapshot[:, i]

        # stats.probplot returns (quantiles, (slope, intercept, r)) and plots
        # onto the given axis; r is the correlation with the Gaussian line
        (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist='norm')
        ax.scatter(osm, osr, s=4, alpha=0.5, color='#4C78A8', zorder=2)
        # diagonal reference line
        x_line = np.array([osm[0], osm[-1]])
        ax.plot(x_line, slope * x_line + intercept,
                color='#E45756', lw=1.5, zorder=3)

        sw_p  = sw_pvals[i]
        color = '#333333' if sw_p > 0.05 else '#E45756'
        ax.set_title(f'Unit {i+1}   r={r:.3f}', fontsize=9, color=color)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('theoretical quantiles', fontsize=7)
        ax.set_ylabel('sample quantiles', fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig2.suptitle(
        f'Q-Q plots at t* = {N_TIMESTEPS-1}  (N={N_TRIALS} trials)\n'
        'Points on the diagonal = Gaussian.  '
        'Red title = Shapiro-Wilk rejects Gaussianity (p < 0.05)',
        fontsize=10
    )
    fig2.tight_layout()
    # fig2.savefig(f'{dir}gaussianity_qq_{SYSTEM}.png',
    #              dpi=300, bbox_inches='tight')

    plt.show()

    end_time = time.time()
    print(f"\nTotal run time: {end_time - start_time:.2f} seconds")
    print(f"Figures saved to {dir}")
