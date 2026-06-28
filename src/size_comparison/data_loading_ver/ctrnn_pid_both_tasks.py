"""
ctrnn_pid_both_tasks.py
=======================
Train a CTRNN on both PerceptualDecisionMaking and ContextDecisionMaking,
run time-resolved Gaussian PID analysis on each, and produce comparison plots.

Mirrors your colleague's notebook structure (load NPZ → train with early
stopping → extract hidden states → PID) but CTRNN only, both tasks, with
the full PID sweep infrastructure from ctrnn_pid_sweep_cdm.py.

Key design decisions
--------------------
- Loads pre-generated NPZ datasets via load_mante_data (same as colleague)
- Trains with validation loss tracking and early stopping (same as colleague)
- CTRNN only (no Elman)
- Both tasks: PDM (input_dim=3) and CDM (input_dim=7)
- Two PID targets per task:
    PDM: signed coherence + binary decision label
    CDM: signed coherence of relevant modality + context label
- Shared test batch used for all PID analysis (never seen during training)
- All plots and raw arrays saved to results/

Usage
-----
    # With pre-generated NPZ datasets
    python ctrnn_pid_both_tasks.py \\
        --pdm_dir data/pdm \\
        --cdm_dir data/cdm \\
        --hidden_size 100

    # Without NPZ (falls back to on-the-fly NeuroGym generation)
    python ctrnn_pid_both_tasks.py --hidden_size 100

    # Quick test run
    python ctrnn_pid_both_tasks.py --pdm_dir data/pdm --cdm_dir data/cdm \\
        --hidden_size 20 --num_epochs 5 --n_test_trials 200

Outputs (results/ctrnn_both_tasks/)
-------------------------------------
    training_curves.png          — loss curves for both tasks
    pid_pdm.png                  — time-resolved PID for PDM
    pid_cdm.png                  — time-resolved PID for CDM
    comparison_4panel.png        — 2x2 grid: task x PID target
    results.npz                  — all raw PID arrays + activations
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import neurogym as ngym

# ──────────────────────────────────────────────────────────────────────────────
# Project module imports
# ──────────────────────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parents[3]))

try:
    from src.models.ctrnn import CTRNN as _ProjectCTRNN
    from src.analysis.gaussian_pid import gaussian_pid_rnn
    from src.tasks.data_loader import load_mante_data
    _USING_PROJECT_MODULES = True
except ImportError as e:
    _USING_PROJECT_MODULES = False
    print(f"[WARN] src/ modules not found ({e}) — using built-in fallbacks.")

print(f"Using project modules: {_USING_PROJECT_MODULES}")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_DIR  = Path('results/ctrnn_both_tasks')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR  = RESULTS_DIR / 'weights'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DT           = 20     # ms per timestep
OUTPUT_DIM   = 3      # fixate / choice1 / choice2
DEFAULT_SEED = 42

# Task specs
TASKS = {
    'pdm': {
        'name':      'PerceptualDecisionMaking-v0',
        'input_dim': 3,
        'timing': {           # fixed timing for PID eval
            'fixation': 200,  # 10 steps
            'stimulus': 2000, # 100 steps
            'delay':    0,
            'decision': 100,  # 5 steps
        },
    },
    'cdm': {
        'name':      'ContextDecisionMaking-v0',
        'input_dim': 7,
        'timing': {
            'fixation': 200,  # 10 steps
            'stimulus': 1000, # 50 steps
            'delay':    0,    # MUST be 0 — default is stochastic
            'decision': 100,  # 5 steps
        },
    },
}

PALETTE = {
    'synergy':    '#54A24B',
    'redundancy': '#4C78A8',
    'unique1':    '#F58518',
    'unique2':    '#E45756',
    'mi_joint':   '#020202',
    'stim_end':   '#CC0000',
    'train':      '#2196F3',
    'val':        '#FF9800',
}

# ──────────────────────────────────────────────────────────────────────────────
# Fallback CTRNN (used when src/models/ctrnn.py is unavailable)
# ──────────────────────────────────────────────────────────────────────────────

class _FallbackCTRNN(nn.Module):
    """Euler CT-RNN: h(t+dt) = h + alpha*(-h + tanh(W_in u + W_rec h))"""
    def __init__(self, input_size, hidden_size, output_size, dt=DT, tau=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = dt / tau
        self.W_in  = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_out = nn.Linear(hidden_size, output_size, bias=True)
        nn.init.orthogonal_(self.W_rec.weight)
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_in.bias)
        nn.init.zeros_(self.W_out.bias)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        outputs, hiddens = [], []
        for t in range(T):
            h = h + self.alpha * (
                -h + torch.tanh(self.W_in(x[:, t]) + self.W_rec(h))
            )
            outputs.append(self.W_out(h))
            hiddens.append(h)
        return torch.stack(outputs, 1), torch.stack(hiddens, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Fallback Gaussian analytic PID (MMI-PID, Barrett 2015)
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_gaussian_pid(activations, target, timestep=None,
                            n_bipartitions=200, seed=DEFAULT_SEED,
                            regularization=1e-5, **kwargs):
    rng = np.random.default_rng(seed)
    N, T, H = activations.shape
    y = target.astype(np.float64)

    def _mi(X):
        X_ = X - X.mean(0)
        y_ = y - y.mean()
        Sxx = X_.T @ X_ / N + regularization * np.eye(X_.shape[1])
        Sxy = X_.T @ y_ / N
        var_y = np.var(y_) + 1e-12
        try:
            r2 = float(Sxy @ np.linalg.solve(Sxx, Sxy)) / var_y
            r2 = np.clip(r2, 0.0, 1 - 1e-9)
        except np.linalg.LinAlgError:
            return 0.0
        return float(-0.5 * np.log(1 - r2) / np.log(2))

    def _pid_at_t(h_t):
        s, r, u1, u2, m = [], [], [], [], []
        half = max(H // 2, 1)
        for _ in range(n_bipartitions):
            idx = rng.permutation(H)
            i1, i2 = idx[:half], idx[half:half*2]
            mi1 = _mi(h_t[:, i1])
            mi2 = _mi(h_t[:, i2])
            mij = _mi(h_t[:, np.concatenate([i1, i2])])
            red = min(mi1, mi2)
            s.append(max(mij - mi1 - mi2 + red, 0.0))
            r.append(red)
            u1.append(max(mi1 - red, 0.0))
            u2.append(max(mi2 - red, 0.0))
            m.append(mij)
        return {k: float(np.mean(v))
                for k, v in zip(['syn','red','u1','u2','mi'], [s,r,u1,u2,m])}

    def _wrap(d):
        return {'synergy': d['syn'], 'redundancy': d['red'],
                'unique1': d['u1'], 'unique2': d['u2'], 'mi_joint': d['mi']}

    if timestep is not None:
        return _wrap(_pid_at_t(activations[:, timestep].astype(np.float64)))

    out = {k: [] for k in ['syn','red','u1','u2','mi']}
    for t in range(T):
        d = _pid_at_t(activations[:, t].astype(np.float64))
        for k in out:
            out[k].append(d[k])
    return {'synergy': np.array(out['syn']), 'redundancy': np.array(out['red']),
            'unique1': np.array(out['u1']),  'unique2':    np.array(out['u2']),
            'mi_joint': np.array(out['mi'])}


# ──────────────────────────────────────────────────────────────────────────────
# Select implementations
# ──────────────────────────────────────────────────────────────────────────────

_CTRNNBase = _ProjectCTRNN  if _USING_PROJECT_MODULES else _FallbackCTRNN
_pid_fn    = gaussian_pid_rnn if _USING_PROJECT_MODULES else _fallback_gaussian_pid


# ──────────────────────────────────────────────────────────────────────────────
# Wrapped CTRNN — uniform (outputs, hidden_states) interface
# ──────────────────────────────────────────────────────────────────────────────

class WrappedCTRNN(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        if _USING_PROJECT_MODULES:
            self.model = _CTRNNBase(input_size=input_dim,
                                    hidden_size=hidden_size,
                                    output_size=OUTPUT_DIM)
        else:
            self.model = _CTRNNBase(input_size=input_dim,
                                    hidden_size=hidden_size,
                                    output_size=OUTPUT_DIM, dt=DT)

    def forward(self, x):
        """x: (B, T, F) → outputs (B,T,3), hidden_states (B,T,H)"""
        if _USING_PROJECT_MODULES:
            outputs, _, hidden_states = self.model(x, return_dynamics=True)
        else:
            outputs, hidden_states = self.model(x)
        return outputs, hidden_states


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def masked_cross_entropy(outputs, targets, periods=None):
    """
    If periods is provided: mask to decision period (periods == 2).
    Otherwise: mask to any non-zero target (targets != 0).
    Matches both colleague's approach (periods==2) and fallback (targets!=0).
    """
    if periods is not None:
        mask = (periods == 2)
    else:
        mask = (targets != 0)

    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=outputs.device)
    return F.cross_entropy(outputs[mask], targets[mask])


# ──────────────────────────────────────────────────────────────────────────────
# Fallback on-the-fly data generator (used when no NPZ provided)
# ──────────────────────────────────────────────────────────────────────────────

def _make_neurogym_generator(task_name, input_dim, batch_size=16, seq_len=130):
    timing = {'delay': 0}
    env = ngym.make(task_name, dt=DT, timing=timing)
    env.reset(seed=DEFAULT_SEED)

    def _gen():
        obs_b, gt_b = [], []
        for _ in range(batch_size):
            env.unwrapped.new_trial()
            ob = env.unwrapped.ob
            gt = env.unwrapped.gt
            T  = ob.shape[0]
            t  = min(T, seq_len)
            ob_p = np.zeros((seq_len, input_dim), dtype=np.float32)
            gt_p = np.zeros((seq_len,), dtype=np.int64)
            ob_p[:t] = ob[:t, :input_dim]
            gt_p[:t] = gt[:t]
            if t < seq_len:
                ob_p[t:] = ob[-1, :input_dim]
                gt_p[t:] = gt[-1]
            obs_b.append(ob_p)
            gt_b.append(gt_p)
        # return (B, T, F) and (B, T)
        return (np.stack(obs_b).astype(np.float32),
                np.stack(gt_b).astype(np.int64))

    return _gen


# ──────────────────────────────────────────────────────────────────────────────
# Training — mirrors colleague's train_rnn with early stopping
# ──────────────────────────────────────────────────────────────────────────────

def train_ctrnn(
    task_key,
    hidden_size,
    device,
    train_loader=None,
    val_loader=None,
    num_epochs=50,
    lr=1e-3,
    patience=10,
    seed=DEFAULT_SEED,
):
    """
    Train a CTRNN with early stopping on validation loss.
    Accepts either a DataLoader (NPZ path provided) or falls back to
    on-the-fly NeuroGym generation.

    Returns model (best weights restored), loss history dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_cfg  = TASKS[task_key]
    input_dim = task_cfg['input_dim']

    model     = WrappedCTRNN(input_dim, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Fallback generator if no DataLoader
    ng_gen = None
    if train_loader is None:
        print(f"  [INFO] No DataLoader provided — using NeuroGym on-the-fly")
        ng_gen = _make_neurogym_generator(task_cfg['name'], input_dim)

    best_val_loss     = float('inf')
    patience_counter  = 0
    best_weights      = copy.deepcopy(model.state_dict())
    history           = {'train_loss': [], 'val_loss': []}

    print(f"\n  Training CTRNN | task={task_key} | hidden={hidden_size} | "
          f"epochs={num_epochs} | patience={patience}")

    for epoch in range(num_epochs):
        # ── Training ──
        model.train()
        train_losses = []

        if train_loader is not None:
            for batch in train_loader:
                obs, labels, periods, cohs, ctxs = batch
                obs     = obs.to(device)
                labels  = labels.to(device)
                periods = periods.to(device)

                optimizer.zero_grad()
                outputs, _ = model(obs)
                loss = masked_cross_entropy(outputs, labels, periods)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

        else:
            # On-the-fly: treat one epoch as n_batches mini-batches
            n_batches = 157
            for _ in range(n_batches):
                obs_np, gt_np = ng_gen()
                obs    = torch.from_numpy(obs_np).to(device)
                labels = torch.from_numpy(gt_np).to(device)

                optimizer.zero_grad()
                outputs, _ = model(obs)
                loss = masked_cross_entropy(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        history['train_loss'].append(avg_train)

        # ── Validation ──
        model.eval()
        val_losses = []

        if val_loader is not None:
            with torch.no_grad():
                for batch in val_loader:
                    obs, labels, periods, cohs, ctxs = batch
                    obs     = obs.to(device)
                    labels  = labels.to(device)
                    periods = periods.to(device)
                    outputs, _ = model(obs)
                    val_losses.append(
                        masked_cross_entropy(outputs, labels, periods).item()
                    )
        else:
            # Validate on a fresh set of on-the-fly trials
            with torch.no_grad():
                for _ in range(20):
                    obs_np, gt_np = ng_gen()
                    obs    = torch.from_numpy(obs_np).to(device)
                    labels = torch.from_numpy(gt_np).to(device)
                    outputs, _ = model(obs)
                    val_losses.append(
                        masked_cross_entropy(outputs, labels).item()
                    )

        avg_val = float(np.mean(val_losses))
        history['val_loss'].append(avg_val)

        # ── Early stopping ──
        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            best_weights     = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1:03d}/{num_epochs}]  "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"patience={patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1} "
                  f"(best val={best_val_loss:.4f})")
            break

    print("    Restoring best weights...")
    model.load_state_dict(best_weights)

    save_path = WEIGHTS_DIR / f'ctrnn_{task_key}_h{hidden_size}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"    Saved → {save_path}")

    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Hidden state extraction — mirrors colleague's extract_hidden_trajectories
# ──────────────────────────────────────────────────────────────────────────────

def extract_hidden_states(model, test_loader, device):
    """
    Run the test set through the model and collect:
        H : (N, T, hidden_size)  hidden states
        Y : (N,)                 signed coherence (PID target 1)
        C : (N,)                 context label    (PID target 2, CDM only)
        P : (N, T)               period labels    (for stim-end detection)
    """
    model.eval()
    H_list, Y_list, C_list, P_list = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            obs, labels, periods, cohs, ctxs = batch
            obs = obs.to(device)

            _, hidden = model(obs)           # (B, T, hidden_size)
            H_list.append(hidden.cpu().numpy())

            # Signed coherence: cohs is (B,) or (B, n_cohs) depending on task
            if cohs.ndim > 1:
                Y_list.append(cohs[:, 0].numpy().astype(float))
            else:
                Y_list.append(cohs.numpy().astype(float))

            C_list.append(ctxs.numpy())
            P_list.append(periods.numpy())

    H = np.concatenate(H_list, axis=0)   # (N, T, H)
    Y = np.concatenate(Y_list, axis=0)   # (N,)
    C = np.concatenate(C_list, axis=0)   # (N,)
    P = np.concatenate(P_list, axis=0)   # (N, T)
    return H, Y, C, P


# ──────────────────────────────────────────────────────────────────────────────
# Stim-end detection
# ──────────────────────────────────────────────────────────────────────────────

def find_stim_end(periods):
    """
    periods : (N, T) int array with 0=fixation, 1=stimulus, 2=decision
    Returns the modal last-stimulus-timestep across trials.
    """
    stim_ends = []
    for p in periods:
        stim_steps = np.where(p == 1)[0]
        stim_ends.append(stim_steps[-1] if len(stim_steps) > 0 else 0)
    return int(np.median(stim_ends))


# ──────────────────────────────────────────────────────────────────────────────
# PID analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_pid_analysis(H, Y, C, P, task_key, n_bipartitions=200):
    """
    Run time-resolved PID for both targets relevant to the task.

    PDM targets: coherence (Y), binary decision (derived from Y sign)
    CDM targets: coherence (Y), context label (C)

    Returns dict with keys per target name.
    """
    stim_end = find_stim_end(P)
    results  = {}

    # Define targets depending on task
    if task_key == 'pdm':
        targets = [
            ('coherence', Y.astype(float)),
            ('decision',  (Y > 0).astype(float)),   # binary: choice1=1, choice2=0
        ]
    else:  # cdm
        targets = [
            ('coherence', Y.astype(float)),
            ('context',   C.astype(float)),
        ]

    for target_name, target_vals in targets:
        print(f"    PID | target={target_name} | "
              f"H={H.shape[2]} | T={H.shape[1]} | N={H.shape[0]}")

        pid_out = _pid_fn(
            activations=H,
            target=target_vals,
            timestep=None,
            n_bipartitions=n_bipartitions,
            seed=DEFAULT_SEED,
            log_base=2,
            regularization=1e-5,
        )

        results[target_name] = {
            'synergy':    np.array(pid_out['synergy']),
            'redundancy': np.array(pid_out['redundancy']),
            'unique1':    np.array(pid_out['unique1']),
            'unique2':    np.array(pid_out['unique2']),
            'mi_joint':   np.array(pid_out['mi_joint']),
            'stim_end':   stim_end,
            'seq_len':    H.shape[1],
        }

        tend = stim_end
        print(f"      stim_end=t{tend} | "
              f"Syn={results[target_name]['synergy'][tend]:.4f}b | "
              f"Red={results[target_name]['redundancy'][tend]:.4f}b")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _plot_pid_ax(ax, pid, title, stim_end=None):
    T    = pid['seq_len']
    t    = np.arange(T)
    tend = stim_end if stim_end is not None else pid['stim_end']

    ax.plot(t, pid['mi_joint'],   label='Total MI',   color=PALETTE['mi_joint'],   lw=2,   ls=':')
    ax.plot(t, pid['synergy'],    label='Synergy',    color=PALETTE['synergy'],    lw=3)
    ax.plot(t, pid['redundancy'], label='Redundancy', color=PALETTE['redundancy'], lw=3)
    ax.plot(t, pid['unique1'],    label='Unique 1',   color=PALETTE['unique1'],    lw=1.5, alpha=0.8)
    ax.plot(t, pid['unique2'],    label='Unique 2',   color=PALETTE['unique2'],    lw=1.5, alpha=0.8)

    ax.axvline(tend, color=PALETTE['stim_end'], ls='--', lw=2,
               label=f'Stim end (t={tend})')
    ax.axvspan(0, tend, color='gray', alpha=0.08)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel(f'Timestep (dt={DT}ms)', fontsize=10)
    ax.set_ylabel('Information (bits)', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)


def plot_training_curves(histories, save_path):
    """Plot train/val loss curves for both tasks side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('CTRNN Training Curves', fontsize=14, fontweight='bold')

    for ax, (task_key, history) in zip(axes, histories.items()):
        epochs = np.arange(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'],
                color=PALETTE['train'], lw=2, label='Train loss')
        ax.plot(epochs, history['val_loss'],
                color=PALETTE['val'],   lw=2, ls='--', label='Val loss')
        ax.set_title(f'{task_key.upper()} Task', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross-entropy loss')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")


def plot_pid_single_task(pid_results, task_key, hidden_size, save_path):
    """Two-panel PID plot for one task (one panel per target)."""
    target_names = list(pid_results.keys())
    fig, axes = plt.subplots(1, len(target_names), figsize=(8 * len(target_names), 5))
    if len(target_names) == 1:
        axes = [axes]

    fig.suptitle(
        f'CTRNN | {task_key.upper()} | hidden={hidden_size}',
        fontsize=13, fontweight='bold', y=1.01
    )

    for ax, tgt in zip(axes, target_names):
        _plot_pid_ax(ax, pid_results[tgt],
                     title=f'PID — target: {tgt}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")


def plot_comparison_4panel(all_pid, hidden_size, save_path):
    """
    2×2 grid matching colleague's notebook layout:
      Row 0: PDM task  (coherence | decision)
      Row 1: CDM task  (coherence | context)
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=False)
    fig.suptitle(
        f'Information Geometry: Task Comparison  (CTRNN, hidden={hidden_size})',
        fontsize=16, fontweight='bold', y=0.98
    )

    panel_map = [
        (0, 0, 'pdm', 'coherence', 'PDM | Coherence target'),
        (0, 1, 'pdm', 'decision',  'PDM | Decision target'),
        (1, 0, 'cdm', 'coherence', 'CDM | Coherence target'),
        (1, 1, 'cdm', 'context',   'CDM | Context target'),
    ]

    # Use shared stim_end reference per row
    stim_end_pdm = all_pid['pdm']['coherence']['stim_end']
    stim_end_cdm = all_pid['cdm']['coherence']['stim_end']
    stim_ends    = {0: stim_end_pdm, 1: stim_end_cdm}

    for row, col, task, tgt, title in panel_map:
        ax  = axes[row, col]
        pid = all_pid[task][tgt]
        _plot_pid_ax(ax, pid, title, stim_end=stim_ends[row])

        if col == 0:
            ax.set_ylabel('Information (bits)', fontsize=11)
        if row == 1:
            ax.set_xlabel(f'Timestep (dt={DT}ms)', fontsize=11)

    # Single legend on first panel
    axes[0, 0].legend(loc='upper left', fontsize=10, framealpha=0.9)
    for row in range(2):
        for col in range(1, 2):
            axes[row, col].get_legend().remove()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(
    hidden_size,
    pdm_dir,
    cdm_dir,
    num_epochs,
    patience,
    lr,
    batch_size,
    n_test_trials,
    n_bipartitions,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device      : {device}")
    if device.type == 'cuda':
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"Hidden size : {hidden_size}")

    # ── Build DataLoaders ──
    loaders = {}
    for task_key, data_dir in [('pdm', pdm_dir), ('cdm', cdm_dir)]:
        if data_dir is not None and _USING_PROJECT_MODULES:
            train_path = os.path.join(data_dir, 'train.npz')
            val_path   = os.path.join(data_dir, 'val.npz')
            test_path  = os.path.join(data_dir, 'test_uniform.npz')
            # fall back to test.npz if test_uniform.npz doesn't exist
            if not os.path.exists(test_path):
                test_path = os.path.join(data_dir, 'test.npz')

            input_dim = TASKS[task_key]['input_dim']
            loaders[task_key] = {
                'train': load_mante_data(train_path, batch_size=batch_size, shuffle=True, input_dim=input_dim),
                'val':   load_mante_data(val_path,   batch_size=batch_size, shuffle=False, input_dim=input_dim),
                'test':  load_mante_data(test_path,  batch_size=batch_size, shuffle=False, input_dim=input_dim),
            }
            print(f"  [{task_key}] Loaded NPZ from {data_dir}")
        else:
            loaders[task_key] = {'train': None, 'val': None, 'test': None}
            print(f"  [{task_key}] No NPZ path — using on-the-fly NeuroGym")

    # ── Train both models ──
    models   = {}
    histories = {}

    for task_key in ('pdm', 'cdm'):
        print(f"\n{'='*60}")
        print(f"  Task: {task_key.upper()}")
        print(f"{'='*60}")

        model, history = train_ctrnn(
            task_key   = task_key,
            hidden_size= hidden_size,
            device     = device,
            train_loader = loaders[task_key]['train'],
            val_loader   = loaders[task_key]['val'],
            num_epochs   = num_epochs,
            lr           = lr,
            patience     = patience,
        )
        models[task_key]    = model
        histories[task_key] = history

    # ── Plot training curves ──
    plot_training_curves(
        histories,
        save_path=RESULTS_DIR / 'training_curves.png'
    )

    # ── Extract hidden states and run PID ──
    all_pid = {}

    for task_key in ('pdm', 'cdm'):
        print(f"\n{'='*60}")
        print(f"  PID Analysis: {task_key.upper()}")
        print(f"{'='*60}")

        model = models[task_key].to('cpu')

        if loaders[task_key]['test'] is not None:
            print(f"  Extracting hidden states from NPZ test set...")
            H, Y, C, P = extract_hidden_states(model, loaders[task_key]['test'], 'cpu')
        else:
            print(f"  Generating {n_test_trials} test trials on-the-fly...")
            task_cfg = TASKS[task_key]
            env = ngym.make(task_cfg['name'], dt=DT, timing=task_cfg['timing'])
            env.reset(seed=DEFAULT_SEED + 999)

            obs_l, gt_l, coh_l, ctx_l, per_l = [], [], [], [], []
            for _ in range(n_test_trials):
                env.unwrapped.new_trial()
                ob    = env.unwrapped.ob.astype(np.float32)
                gt    = env.unwrapped.gt.astype(np.int64)
                trial = env.unwrapped.trial

                # Build period array from gt
                period = np.zeros(len(gt), dtype=np.int8)
                fix_steps  = np.where(ob[:, 0] > 0.5)[0]
                stim_steps = np.where((ob[:, 0] < 0.5) & (gt == 0))[0]
                dec_steps  = np.where(gt != 0)[0]
                period[fix_steps]  = 0
                period[stim_steps] = 1
                period[dec_steps]  = 2

                # Coherence
                if task_key == 'pdm':
                    coh = float(trial.get('coh', 0.0))
                    gt_label = int(trial.get('ground_truth', 0))
                    signed_coh = coh if gt_label == 0 else -coh
                    ctx = 0
                else:
                    context   = int(trial.get('context', 0))
                    coh_0     = float(trial.get('coh_0', trial.get('coh_1', 0.0)))
                    coh_1     = float(trial.get('coh_1', trial.get('coh_2', 0.0)))
                    rel_coh   = coh_0 if context == 0 else coh_1
                    gt_label  = int(trial.get('ground_truth', 1))
                    sign      = 1 if gt_label == 1 else -1
                    signed_coh = sign * rel_coh
                    ctx = context

                obs_l.append(ob)
                gt_l.append(gt)
                coh_l.append(signed_coh)
                ctx_l.append(ctx)
                per_l.append(period)

            # Pad all trials to the same length
            max_T = max(o.shape[0] for o in obs_l)
            F     = obs_l[0].shape[1]
            obs_pad = np.zeros((n_test_trials, max_T, F), dtype=np.float32)
            gt_pad  = np.zeros((n_test_trials, max_T), dtype=np.int64)
            per_pad = np.zeros((n_test_trials, max_T), dtype=np.int8)
            for i, (ob, gt, per) in enumerate(zip(obs_l, gt_l, per_l)):
                T = ob.shape[0]
                obs_pad[i, :T] = ob
                gt_pad[i,  :T] = gt
                per_pad[i, :T] = per
                if T < max_T:
                    obs_pad[i, T:] = ob[-1]
                    gt_pad[i,  T:] = gt[-1]
                    per_pad[i, T:] = per[-1]

            obs_t = torch.from_numpy(obs_pad)
            model.eval()
            with torch.no_grad():
                _, h_seq = model(obs_t)
            H = h_seq.numpy()
            Y = np.array(coh_l, dtype=np.float64)
            C = np.array(ctx_l, dtype=np.int32)
            P = per_pad

        print(f"  H shape: {H.shape}  Y range: [{Y.min():.1f}, {Y.max():.1f}]")

        pid_results = run_pid_analysis(H, Y, C, P, task_key, n_bipartitions)
        all_pid[task_key] = pid_results

        # Per-task plot
        plot_pid_single_task(
            pid_results, task_key, hidden_size,
            save_path=RESULTS_DIR / f'pid_{task_key}.png'
        )

    # ── 4-panel comparison ──
    plot_comparison_4panel(
        all_pid, hidden_size,
        save_path=RESULTS_DIR / 'comparison_4panel.png'
    )

    # ── Save raw results ──
    npz_path = RESULTS_DIR / 'results.npz'
    save_dict = {}
    for task_key, pid_results in all_pid.items():
        for tgt, data in pid_results.items():
            for atom in ('synergy','redundancy','unique1','unique2','mi_joint'):
                save_dict[f'{task_key}_{tgt}_{atom}'] = data[atom]
            save_dict[f'{task_key}_{tgt}_stim_end'] = np.array(data['stim_end'])
    for task_key, history in histories.items():
        save_dict[f'{task_key}_train_loss'] = np.array(history['train_loss'])
        save_dict[f'{task_key}_val_loss']   = np.array(history['val_loss'])

    np.savez_compressed(str(npz_path), **save_dict)
    print(f"\nRaw results → {npz_path}")
    print("Done.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Train CTRNN on PDM + CDM and run Gaussian PID analysis')
    p.add_argument('--hidden_sizes', nargs='+', type=int, default=[100],
               help='One or more hidden sizes, e.g. --hidden_sizes 20 50 100')
    p.add_argument('--pdm_dir',        type=str,   default=None,
                   help='Directory containing PDM train/val/test NPZ files')
    p.add_argument('--cdm_dir',        type=str,   default=None,
                   help='Directory containing CDM train/val/test NPZ files')
    p.add_argument('--num_epochs',     type=int,   default=50)
    p.add_argument('--patience',       type=int,   default=10)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--batch_size',     type=int,   default=2048)
    p.add_argument('--n_test_trials',  type=int,   default=2000)
    p.add_argument('--n_bipartitions', type=int,   default=200)
    args = p.parse_args()

for h in args.hidden_sizes:
    print(f"\n{'#'*60}")
    print(f"  HIDDEN SIZE = {h}")
    print(f"{'#'*60}")
    
    main(
        hidden_size    = h,
        pdm_dir        = args.pdm_dir,
        cdm_dir        = args.cdm_dir,
        num_epochs     = args.num_epochs,
        patience       = args.patience,
        lr             = args.lr,
        batch_size     = args.batch_size,
        n_test_trials  = args.n_test_trials,
        n_bipartitions = args.n_bipartitions,
    )