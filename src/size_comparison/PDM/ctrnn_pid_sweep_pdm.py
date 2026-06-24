"""
ctrnn_pid_sweep.py
==================
Train a CTRNN on PerceptualDecisionMaking-v0 across hidden sizes [20, 16, 12, 8, 4, 2],
run time-resolved Gaussian PID analysis at each size, save all plots, and produce a
summary comparison figure showing how PID structure changes with network capacity.

Usage
-----
    python ctrnn_pid_sweep.py
    python ctrnn_pid_sweep.py --hidden_sizes 20 10 5 2
    python ctrnn_pid_sweep.py --n_train_batches 1000 --n_test_trials 200

Outputs (all saved to results/pid_sweep/)
-----------------------------------------
    pid_timeseries_h{N}.png   — time-resolved PID plot for each hidden size
    comparison_summary.png    — multi-panel comparison across all hidden sizes
    sweep_results.npz         — raw PID numbers for further analysis
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import neurogym as ngym

# ──────────────────────────────────────────────────────────────────────────────
# Paths — adjust if your project layout differs
# ──────────────────────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))    # for src/models/ctrnn.py and src/analysis/gaussian_pid.py

try:
    from src.models.ctrnn import CTRNN
    from src.analysis.gaussian_pid import gaussian_pid_rnn
    from src.tasks.neurogym_wrapper import create_dataset_generator
    _USING_PROJECT_MODULES = True
except ImportError:
    _USING_PROJECT_MODULES = False
    print("[WARN] src modules not found — using built-in fallbacks (see code).")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

TASK          = 'PerceptualDecisionMaking-v0'
INPUT_DIM     = 3          # fixation + 2 stimulus channels
OUTPUT_DIM    = 3          # fixate / choice1 / choice2
DT            = 20         # ms per timestep
RESULTS_DIR   = 'results/pid_sweep'

DEFAULT_HIDDEN_SIZES  = [40, 60, 80, 100, 150, 20, 16, 12, 8, 4, 2]
DEFAULT_N_TRAIN       = 800    # batches (each batch = 16 trials)
DEFAULT_N_TEST        = 200    # trials for PID analysis
DEFAULT_LR            = 2e-3
DEFAULT_BATCH_SIZE    = 16
DEFAULT_N_BIPARTITIONS = 50
DEFAULT_SEED          = 42

# Fixed timing for PID evaluation — keeps sequences identical across runs
EVAL_TIMING = {
    'fixation': 200,    # 10 steps  @ dt=20
    'stimulus': 1000,   # 50 steps
    'delay':    0,
    'decision': 100,    # 5  steps
}
# Total = 65 steps → seq_len = 65

PALETTE = {
    'synergy':    '#54A24B',
    'redundancy': '#4C78A8',
    'unique1':    '#F58518',
    'unique2':    '#E45756',
    'mi_joint':   '#000000',
    'stim_end':   '#CC0000',
}

# ──────────────────────────────────────────────────────────────────────────────
# Fallback implementations (used when src/ modules are unavailable)
# ──────────────────────────────────────────────────────────────────────────────

class _FallbackCTRNN(nn.Module):
    """
    Minimal CT-RNN (Euler discretisation).
    h(t+dt) = h(t) + (dt/tau)*(-h(t) + tanh(W_rec h(t) + W_in u(t) + b))
    """
    def __init__(self, input_size, hidden_size, output_size,
                 dt=20, tau=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = dt / tau                         # Euler step fraction

        self.W_in  = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_out = nn.Linear(hidden_size, output_size, bias=True)

        # Initialisation
        nn.init.orthogonal_(self.W_rec.weight)
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_in.bias)
        nn.init.zeros_(self.W_out.bias)

    def forward(self, x):
        """
        x : (batch, seq_len, input_size)
        returns
            outputs     : (batch, seq_len, output_size)
            hidden_states : (batch, seq_len, hidden_size)
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        outputs, hiddens = [], []

        for t in range(T):
            h = h + self.alpha * (
                -h + torch.tanh(self.W_in(x[:, t, :]) + self.W_rec(h))
            )
            outputs.append(self.W_out(h))
            hiddens.append(h)

        outputs = torch.stack(outputs, dim=1)         # (B, T, out)
        hiddens = torch.stack(hiddens, dim=1)         # (B, T, H)
        return outputs, hiddens


def _fallback_gaussian_pid(activations, target,
                            timestep=None, n_bipartitions=50, seed=42,
                            regularization=1e-5, **kwargs):
    """
    Gaussian analytic PID (MMI-PID, Barrett 2015).

    activations : (n_trials, seq_len, hidden_size)
    target      : (n_trials,)  continuous scalar (signed coherence or binary choice)
    timestep    : int | None — if None, compute for all timesteps
    returns dict with keys: synergy, redundancy, unique1, unique2, mi_joint
                    each an array of length seq_len (or scalar if timestep given)
    """
    rng = np.random.default_rng(seed)
    N, T, H = activations.shape
    y = target.astype(np.float64)                     # (N,)

    def _gaussian_mi(X, Y):
        """
        MI between multivariate Gaussian X (N,d) and scalar Y (N,).
        I(X;Y) = 0.5 * log( det(Sigma_X) / det(Sigma_X|Y) )
               = 0.5 * log( var(Y) / var(Y|X) )  when Y is scalar
        Uses: I = 0.5 * log det(Sigma_X) - 0.5 * log det(Sigma_X|Y)
        For scalar Y: I(X;Y) = H(Y) - H(Y|X)
                              = 0.5*log(2πe var(Y)) - 0.5*log(2πe var(Y|X))
        We use the regression formula: var(Y|X) = var(Y)(1 - R²)
        """
        # Add jitter for numerical stability
        X_ = X - X.mean(0)
        y_ = Y - Y.mean()
        Sxx = X_.T @ X_ / N + regularization * np.eye(X_.shape[1])
        Sxy = X_.T @ y_ / N                          # (d,)
        # R² = Sxy' Sxx^{-1} Sxy / var(y)
        var_y = np.var(y_) + 1e-12
        try:
            Sxx_inv_Sxy = np.linalg.solve(Sxx, Sxy)
            r2 = float(Sxy @ Sxx_inv_Sxy) / var_y
            r2 = np.clip(r2, 0.0, 1.0 - 1e-9)
        except np.linalg.LinAlgError:
            return 0.0
        # I(X;Y) = -0.5 * log(1 - R²)  [nats]; convert to bits
        return float(-0.5 * np.log(1.0 - r2) / np.log(2))

    def _mmi_pid_at_t(h_t):
        """
        Compute MMI-PID for one timestep.
        h_t : (N, H) hidden states
        Splits H units into two halves (X1, X2) and computes
        Red = min(I(X1;Y), I(X2;Y))
        Joint MI = I(X1,X2;Y)
        Syn = Joint - I(X1;Y) - I(X2;Y) + Red
        Unq1 = I(X1;Y) - Red
        Unq2 = I(X2;Y) - Red
        Averaged over n_bipartitions random 50/50 splits.
        """
        results = {'syn': [], 'red': [], 'u1': [], 'u2': [], 'mi': []}
        half = max(H // 2, 1)

        for _ in range(n_bipartitions):
            idx = rng.permutation(H)
            i1, i2 = idx[:half], idx[half:half*2]
            X1 = h_t[:, i1]
            X2 = h_t[:, i2]
            Xj = h_t[:, np.concatenate([i1, i2])]

            mi1  = _gaussian_mi(X1, y)
            mi2  = _gaussian_mi(X2, y)
            mi_j = _gaussian_mi(Xj, y)

            red  = min(mi1, mi2)
            syn  = max(mi_j - mi1 - mi2 + red, 0.0)
            u1   = max(mi1 - red, 0.0)
            u2   = max(mi2 - red, 0.0)

            results['syn'].append(syn)
            results['red'].append(red)
            results['u1'].append(u1)
            results['u2'].append(u2)
            results['mi'].append(mi_j)

        return {k: float(np.mean(v)) for k, v in results.items()}

    if timestep is not None:
        r = _mmi_pid_at_t(activations[:, timestep, :].astype(np.float64))
        return {
            'synergy':    r['syn'],
            'redundancy': r['red'],
            'unique1':    r['u1'],
            'unique2':    r['u2'],
            'mi_joint':   r['mi'],
        }

    # All timesteps
    syn_t, red_t, u1_t, u2_t, mi_t = [], [], [], [], []
    for t in range(T):
        r = _mmi_pid_at_t(activations[:, t, :].astype(np.float64))
        syn_t.append(r['syn'])
        red_t.append(r['red'])
        u1_t.append(r['u1'])
        u2_t.append(r['u2'])
        mi_t.append(r['mi'])

    return {
        'synergy':    np.array(syn_t),
        'redundancy': np.array(red_t),
        'unique1':    np.array(u1_t),
        'unique2':    np.array(u2_t),
        'mi_joint':   np.array(mi_t),
    }


def _fallback_dataset_generator(task, dt=DT, batch_size=DEFAULT_BATCH_SIZE,
                                 seq_len=120):
    """Returns a callable that yields (inputs_np, targets_np) batches."""
    env = ngym.make(task, dt=dt)
    env.reset(seed=DEFAULT_SEED)

    def _generator():
        obs_batch, gt_batch = [], []
        for _ in range(batch_size):
            env.unwrapped.new_trial()
            ob = env.unwrapped.ob      # (T, ob_size)
            gt = env.unwrapped.gt      # (T,)
            T  = ob.shape[0]
            t  = min(T, seq_len)
            # pad
            ob_pad = np.zeros((seq_len, ob.shape[1]), dtype=np.float32)
            gt_pad = np.zeros((seq_len,), dtype=np.int64)
            ob_pad[:t] = ob[:t]
            gt_pad[:t] = gt[:t]
            if t < seq_len:
                ob_pad[t:] = ob[-1]
                gt_pad[t:] = gt[-1]
            obs_batch.append(ob_pad)
            gt_batch.append(gt_pad)
        # Return (seq_len, batch, features) to match NeuroGym convention
        obs = np.stack(obs_batch, axis=1)   # (T, B, F)
        tgt = np.stack(gt_batch,  axis=1)   # (T, B)
        return obs, tgt

    return _generator


# ──────────────────────────────────────────────────────────────────────────────
# Select implementations
# ──────────────────────────────────────────────────────────────────────────────

if _USING_PROJECT_MODULES:
    CTRNNImpl       = CTRNN
    pid_fn          = gaussian_pid_rnn
    make_dataset    = create_dataset_generator
else:
    CTRNNImpl       = _FallbackCTRNN
    pid_fn          = _fallback_gaussian_pid
    make_dataset    = _fallback_dataset_generator


# ──────────────────────────────────────────────────────────────────────────────
# Wrapped CTRNN
# ──────────────────────────────────────────────────────────────────────────────

class WrappedCTRNN(nn.Module):
    """
    Thin wrapper so the CTRNN always returns (outputs, hidden_states)
    in (batch, seq, features) layout regardless of which implementation is used.
    """
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        if _USING_PROJECT_MODULES:
            self.model = CTRNNImpl(
                input_size=input_dim,
                hidden_size=hidden_size,
                output_size=OUTPUT_DIM,
            )
        else:
            self.model = CTRNNImpl(
                input_size=input_dim,
                hidden_size=hidden_size,
                output_size=OUTPUT_DIM,
                dt=DT,
            )

    def forward(self, x):
        """x : (batch, seq, input_dim) → outputs, hidden_states both (batch, seq, *)"""
        if _USING_PROJECT_MODULES:
            outputs, _, hidden_states = self.model(x, return_dynamics=True)
        else:
            outputs, hidden_states = self.model(x)
        return outputs, hidden_states


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def masked_cross_entropy(outputs, targets):
    """Cross-entropy only on decision timesteps (target != 0)."""
    mask = targets != 0
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=outputs.device)
    return F.cross_entropy(outputs[mask], targets[mask])


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_ctrnn(hidden_size, n_batches, device, weights_dir, seed=DEFAULT_SEED):
    """Train one CTRNN and return the trained model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = WrappedCTRNN(INPUT_DIM, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LR)
    dataset   = make_dataset(TASK)

    print(f"\n  Training CTRNN  hidden={hidden_size}  batches={n_batches}")
    losses = []

    for i in range(n_batches):
        inputs_np, targets_np = dataset()

        # Dataset generator returns (T, B, F) — convert to (B, T, F)
        inputs  = torch.from_numpy(inputs_np).float().transpose(0, 1).to(device)
        targets = torch.from_numpy(targets_np).long().transpose(0, 1).to(device)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = masked_cross_entropy(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if (i + 1) % 200 == 0 or (i + 1) == n_batches:
            print(f"    batch {i+1:>4}/{n_batches}  loss={np.mean(losses[-50:]):.4f}")

    # Save weights
    save_path = os.path.join(weights_dir, f"ctrnn_h{hidden_size}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"    Saved → {save_path}")
    return model, losses


# ──────────────────────────────────────────────────────────────────────────────
# Test batch generation
# ──────────────────────────────────────────────────────────────────────────────

def get_test_batch(n_trials, device, seed=DEFAULT_SEED):
    """
    Generate a fixed test batch using deterministic timing.
    Returns inputs (B,T,F) and targets (B,T) as torch tensors on device,
    plus signed_coherences (B,) as numpy for PID target.
    """
    env = ngym.make(TASK, dt=DT, timing=EVAL_TIMING)
    env.reset(seed=seed)

    obs_list, gt_list, coh_list = [], [], []

    for _ in range(n_trials):
        env.unwrapped.new_trial()
        ob  = env.unwrapped.ob.astype(np.float32)   # (T, 3)
        gt  = env.unwrapped.gt.astype(np.int64)      # (T,)
        trial = env.unwrapped.trial
        coh   = float(trial.get('coh', 0.0))
        gtruth = int(trial.get('ground_truth', 0))
        signed_coh = coh if gtruth == 0 else -coh    # 0→choice1, 1→choice2

        T  = ob.shape[0]
        obs_list.append(ob)
        gt_list.append(gt)
        coh_list.append(signed_coh)

    # All trials same length with fixed timing
    inputs  = torch.tensor(np.stack(obs_list),  dtype=torch.float32).to(device)
    targets = torch.tensor(np.stack(gt_list),   dtype=torch.long).to(device)
    cohs    = np.array(coh_list, dtype=np.float32)

    return inputs, targets, cohs


# ──────────────────────────────────────────────────────────────────────────────
# PID analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_pid(model, inputs, targets, cohs, n_bipartitions, device):
    """
    Run time-resolved Gaussian PID on a trained model.
    Returns dict with PID arrays and metadata.
    """
    model.eval()
    with torch.no_grad():
        _, h_seq = model(inputs)             # (B, T, H)
    acts = h_seq.cpu().numpy()              # (B, T, H)

    # Use signed coherence as continuous PID target (more informative than binary)
    # Fall back to binary decision if coherence is all zero
    if np.std(cohs) < 1e-6:
        pid_target = targets[:, -1].cpu().numpy().astype(float)
        target_label = 'binary decision'
    else:
        pid_target = cohs.astype(float)
        target_label = 'signed coherence'

    print(f"    PID target: {target_label}  |  H={acts.shape[2]}  T={acts.shape[1]}")

    pid_out = pid_fn(
        activations=acts,
        target=pid_target,
        timestep=None,
        n_bipartitions=n_bipartitions,
        seed=DEFAULT_SEED,
        regularization=1e-5,
    )

    # Find stimulus-end timestep from fixation channel
    fix_channel  = inputs[:, :, 0].cpu().numpy()  # (B, T)
    stim_ends = np.array([
        np.where(fix_channel[b] < 0.5)[0][-1]
        for b in range(fix_channel.shape[0])])
    avg_stim_end = int(np.median(stim_ends))

    return {
        'synergy':    np.array(pid_out['synergy']),
        'redundancy': np.array(pid_out['redundancy']),
        'unique1':    np.array(pid_out['unique1']),
        'unique2':    np.array(pid_out['unique2']),
        'mi_joint':   np.array(pid_out['mi_joint']),
        'stim_end':   avg_stim_end,
        'seq_len':    acts.shape[1],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting — per-model time series
# ──────────────────────────────────────────────────────────────────────────────

def plot_timeseries(pid_data, hidden_size, loss_curve, save_path):
    """Two-panel plot: loss curve + time-resolved PID."""
    fig, (ax_loss, ax_pid) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={'width_ratios': [1, 2.5]}
    )
    fig.suptitle(
        f'CTRNN  hidden={hidden_size}  —  Training & PID Analysis',
        fontsize=13, fontweight='bold', y=1.01
    )

    # ── Loss curve ──
    smooth_loss = np.convolve(loss_curve, np.ones(20)/20, mode='valid')
    ax_loss.plot(loss_curve, alpha=0.25, color='steelblue', linewidth=0.8)
    ax_loss.plot(np.arange(19, len(loss_curve)), smooth_loss,
                 color='steelblue', linewidth=2, label='Loss (20-batch avg)')
    ax_loss.set_xlabel('Training batch')
    ax_loss.set_ylabel('Masked cross-entropy loss')
    ax_loss.set_title('Training loss')
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)

    # ── PID time series ──
    T    = pid_data['seq_len']
    t    = np.arange(T)
    tend = pid_data['stim_end']

    ax_pid.plot(t, pid_data['mi_joint'],   label='Total MI',    color=PALETTE['mi_joint'],   lw=2.5, ls=':')
    ax_pid.plot(t, pid_data['synergy'],    label='Synergy',     color=PALETTE['synergy'],    lw=2.5)
    ax_pid.plot(t, pid_data['redundancy'], label='Redundancy',  color=PALETTE['redundancy'], lw=2.5)
    ax_pid.plot(t, pid_data['unique1'],    label='Unique 1',    color=PALETTE['unique1'],    lw=1.5, alpha=0.8)
    ax_pid.plot(t, pid_data['unique2'],    label='Unique 2',    color=PALETTE['unique2'],    lw=1.5, alpha=0.8)

    ax_pid.axvline(tend, color=PALETTE['stim_end'], ls='--', lw=1.5, label=f'Stim end (t={tend})')
    ax_pid.axvspan(0, tend, color='gray', alpha=0.08)

    ax_pid.set_xlabel(f'Timestep (dt={DT}ms)')
    ax_pid.set_ylabel('Information (bits)')
    ax_pid.set_title('Time-resolved Gaussian PID')
    ax_pid.legend(fontsize=9, loc='upper left')
    ax_pid.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting — comparison summary
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison(all_results, save_path):
    """
    Multi-panel summary comparing all hidden sizes:
      Row 1: time-resolved synergy across all sizes (one line per size)
      Row 2: time-resolved redundancy across all sizes
      Row 3: bar charts of PID atoms at stimulus-end snapshot
      Row 4: final loss vs hidden size + synergy@stim_end vs hidden size
    """
    hidden_sizes = sorted(all_results.keys(), reverse=True)
    n            = len(hidden_sizes)
    cmap         = plt.cm.viridis(np.linspace(0.15, 0.85, n))

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_syn  = fig.add_subplot(gs[0, :])
    ax_red  = fig.add_subplot(gs[1, :])
    ax_bar  = fig.add_subplot(gs[2, 0])
    ax_pi   = fig.add_subplot(gs[2, 1])
    ax_loss = fig.add_subplot(gs[3, 0])
    ax_cap  = fig.add_subplot(gs[3, 1])

    fig.suptitle(
        'CTRNN PID Sweep — Hidden Size Comparison\n'
        'PerceptualDecisionMaking-v0',
        fontsize=14, fontweight='bold'
    )

    stim_end_ref = all_results[hidden_sizes[0]]['pid']['stim_end']

    # ── Row 1: Synergy over time ──
    for i, h in enumerate(hidden_sizes):
        r   = all_results[h]
        pid = r['pid']
        T   = pid['seq_len']
        ax_syn.plot(np.arange(T), pid['synergy'],
                    color=cmap[i], lw=2, label=f'H={h}')
    ax_syn.axvline(stim_end_ref, color='red', ls='--', lw=1, alpha=0.6,
                   label='Stim end')
    ax_syn.set_title('Synergy over time — all hidden sizes')
    ax_syn.set_ylabel('Synergy (bits)')
    ax_syn.set_xlabel(f'Timestep (dt={DT}ms)')
    ax_syn.legend(ncol=n, fontsize=8, loc='upper left')
    ax_syn.grid(alpha=0.25)

    # ── Row 2: Redundancy over time ──
    for i, h in enumerate(hidden_sizes):
        pid = all_results[h]['pid']
        T   = pid['seq_len']
        ax_red.plot(np.arange(T), pid['redundancy'],
                    color=cmap[i], lw=2, label=f'H={h}')
    ax_red.axvline(stim_end_ref, color='red', ls='--', lw=1, alpha=0.6)
    ax_red.set_title('Redundancy over time — all hidden sizes')
    ax_red.set_ylabel('Redundancy (bits)')
    ax_red.set_xlabel(f'Timestep (dt={DT}ms)')
    ax_red.legend(ncol=n, fontsize=8, loc='upper left')
    ax_red.grid(alpha=0.25)

    # ── Row 3 left: PID atoms at stim-end, grouped bar ──
    atoms    = ['synergy', 'redundancy', 'unique1', 'unique2']
    atom_colors = [PALETTE['synergy'], PALETTE['redundancy'],
                   PALETTE['unique1'], PALETTE['unique2']]
    x        = np.arange(len(hidden_sizes))
    width    = 0.18
    offsets  = np.linspace(-0.27, 0.27, len(atoms))

    for j, (atom, color) in enumerate(zip(atoms, atom_colors)):
        vals = [all_results[h]['pid'][atom][all_results[h]['pid']['stim_end']]
                for h in hidden_sizes]
        ax_bar.bar(x + offsets[j], vals, width, label=atom.capitalize(),
                   color=color, alpha=0.85)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f'H={h}' for h in hidden_sizes], fontsize=9)
    ax_bar.set_title('PID atoms at stimulus-end snapshot')
    ax_bar.set_ylabel('Information (bits)')
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis='y', alpha=0.3)

    # ── Row 3 right: Synergy / (Synergy + Redundancy) ratio at stim-end ──
    ratios = []
    for h in hidden_sizes:
        pid  = all_results[h]['pid']
        tend = pid['stim_end']
        syn  = pid['synergy'][tend]
        red  = pid['redundancy'][tend]
        denom = syn + red
        ratios.append(syn / denom if denom > 1e-10 else 0.0)

    bar_colors = [cmap[i] for i in range(n)]
    ax_pi.bar([f'H={h}' for h in hidden_sizes], ratios,
              color=bar_colors, alpha=0.85)
    ax_pi.axhline(0.5, color='gray', ls='--', lw=1, label='Syn = Red')
    ax_pi.set_ylim(0, 1)
    ax_pi.set_title('Synergy ratio at stim-end\n[Syn / (Syn + Red)]')
    ax_pi.set_ylabel('Synergy ratio')
    ax_pi.set_xlabel('Hidden size')
    ax_pi.legend(fontsize=8)
    ax_pi.grid(axis='y', alpha=0.3)

    # ── Row 4 left: Final loss vs hidden size ──
    final_losses = [np.mean(all_results[h]['losses'][-50:])
                    for h in hidden_sizes]
    ax_loss.plot([str(h) for h in hidden_sizes], final_losses,
                 'o-', color='steelblue', lw=2, ms=8)
    ax_loss.set_title('Final training loss vs hidden size')
    ax_loss.set_xlabel('Hidden size')
    ax_loss.set_ylabel('Loss (last 50 batches avg)')
    ax_loss.grid(alpha=0.3)

    # ── Row 4 right: Synergy and Redundancy at stim-end vs hidden size ──
    syn_at_end = [all_results[h]['pid']['synergy'][all_results[h]['pid']['stim_end']]
                  for h in hidden_sizes]
    red_at_end = [all_results[h]['pid']['redundancy'][all_results[h]['pid']['stim_end']]
                  for h in hidden_sizes]

    ax_cap.plot([str(h) for h in hidden_sizes], syn_at_end,
                'o-', color=PALETTE['synergy'],    lw=2, ms=8, label='Synergy')
    ax_cap.plot([str(h) for h in hidden_sizes], red_at_end,
                's-', color=PALETTE['redundancy'], lw=2, ms=8, label='Redundancy')
    ax_cap.set_title('Synergy & Redundancy at stim-end vs hidden size')
    ax_cap.set_xlabel('Hidden size')
    ax_cap.set_ylabel('Information (bits)')
    ax_cap.legend(fontsize=9)
    ax_cap.grid(alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison figure saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(hidden_sizes, n_train_batches, n_test_trials):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Training batches per model: {n_train_batches}")
    print(f"Test trials for PID: {n_test_trials}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    weights_dir = os.path.join(RESULTS_DIR, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # Generate one shared test batch — same trials evaluated for all models
    print("\nGenerating fixed test batch...")
    inputs, targets, cohs = get_test_batch(n_test_trials, device)
    print(f"  inputs: {tuple(inputs.shape)}  "
          f"coherence range: [{cohs.min():.1f}, {cohs.max():.1f}]")

    all_results = {}

    for h in hidden_sizes:
        print(f"\n{'='*55}")
        print(f"  Hidden size = {h}")
        print(f"{'='*55}")

        # Train
        model, losses = train_ctrnn(h, n_train_batches, device, weights_dir)

        # PID
        n_bipartitions = 2 * h
        print(f"  Running PID analysis (n_bipartitions={n_bipartitions})...")
        pid_data = run_pid(model, inputs, targets, cohs, n_bipartitions, device)

        print(f"  Stim-end t={pid_data['stim_end']} | "
              f"Syn={pid_data['synergy'][pid_data['stim_end']]:.4f} bits | "
              f"Red={pid_data['redundancy'][pid_data['stim_end']]:.4f} bits")

        all_results[h] = {'pid': pid_data, 'losses': losses}

        # Per-model plot
        ts_path = os.path.join(RESULTS_DIR, f'pid_timeseries_h{h}.png')
        plot_timeseries(pid_data, h, losses, ts_path)

    # Summary comparison
    comp_path = os.path.join(RESULTS_DIR, 'comparison_summary.png')
    plot_comparison(all_results, comp_path)

    # Save raw numbers
    npz_path = os.path.join(RESULTS_DIR, 'sweep_results.npz')
    save_dict = {}
    for h, r in all_results.items():
        for atom in ('synergy', 'redundancy', 'unique1', 'unique2', 'mi_joint'):
            save_dict[f'h{h}_{atom}'] = r['pid'][atom]
        save_dict[f'h{h}_losses']   = np.array(r['losses'])
        save_dict[f'h{h}_stim_end'] = np.array(r['pid']['stim_end'])
    np.savez_compressed(npz_path, **save_dict)
    print(f"Raw results saved → {npz_path}")

    print("\nDone.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='CTRNN PID sweep over hidden sizes')
    p.add_argument('--hidden_sizes',     nargs='+', type=int,
                   default=DEFAULT_HIDDEN_SIZES)
    p.add_argument('--n_train_batches',  type=int, default=DEFAULT_N_TRAIN)
    p.add_argument('--n_test_trials',    type=int, default=DEFAULT_N_TEST)
    args = p.parse_args()

    main(
        hidden_sizes   = args.hidden_sizes,
        n_train_batches= args.n_train_batches,
        n_test_trials  = args.n_test_trials,
    )