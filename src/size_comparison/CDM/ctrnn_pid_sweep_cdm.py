"""
ctrnn_pid_sweep_cdm.py
======================
Train a CTRNN on ContextDecisionMaking-v0 across hidden sizes [20, 16, 12, 8, 4, 2],
run time-resolved Gaussian PID analysis at each size, save all plots, and produce a
summary comparison figure.

Key differences from the PDM version
--------------------------------------
- ob_size = 7  (fixation + 4 stimulus channels + 2 context channels)
- Coherences are [5, 15, 50] not Britten levels
- Stochastic delay period → MUST be fixed for PID (set delay=0)
- Two PID targets analysed:
    (a) signed coherence of the RELEVANT modality  (task difficulty + direction)
    (b) context label (0 or 1)                     (which rule is active)
- Stim-end detection uses fixation channel (ch0) same as PDM

Observation vector layout (ob_size = 7)
-----------------------------------------
  ch 0 : fixation signal
  ch 1 : stim1_mod1  (relevant or irrelevant depending on context)
  ch 2 : stim2_mod1
  ch 3 : stim1_mod2
  ch 4 : stim2_mod2
  ch 5 : context1 cue  (=1 when context==0, i.e. attend modality 1)
  ch 6 : context2 cue  (=1 when context==1, i.e. attend modality 2)

Usage
-----
    python ctrnn_pid_sweep_cdm.py
    python ctrnn_pid_sweep_cdm.py --hidden_sizes 20 10 5 2
    python ctrnn_pid_sweep_cdm.py --n_train_batches 1000 --n_test_trials 200

Outputs (results/pid_sweep_cdm/)
---------------------------------
    pid_timeseries_h{N}.png    — per hidden size: loss + two PID panels
    comparison_summary.png     — multi-panel sweep comparison
    sweep_results.npz          — raw arrays for further analysis
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
# Project module imports (with fallback)
# ──────────────────────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))    

try:
    from src.models.ctrnn import CTRNN
    from src.analysis.gaussian_pid import gaussian_pid_rnn
    from src.tasks.neurogym_wrapper import create_dataset_generator
    _USING_PROJECT_MODULES = True
except ImportError:
    _USING_PROJECT_MODULES = False
    print("[WARN] src/ modules not found — using built-in fallbacks.")

print(f"Using project modules: {_USING_PROJECT_MODULES}")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

TASK        = 'ContextDecisionMaking-v0'
INPUT_DIM   = 7     # fixation + 4 stim + 2 context channels
OUTPUT_DIM  = 3     # fixate / choice1 / choice2
DT          = 20    # ms per timestep
RESULTS_DIR = 'results/pid_sweep_cdm'

DEFAULT_HIDDEN_SIZES   = [2, 4, 8, 12, 16, 20, 40, 60, 80, 100, 150, 200]
DEFAULT_N_TRAIN        = 2500
DEFAULT_N_TEST         = 500
DEFAULT_LR             = 2e-3
DEFAULT_BATCH_SIZE     = 16
DEFAULT_N_BIPARTITIONS = 50
DEFAULT_SEED           = 42

# Fixed timing for PID evaluation
# CRITICAL: delay must be 0 — default is stochastic TruncExp(600,300,3000)
# which makes trials different lengths and breaks time-aligned PID.
EVAL_TIMING = {
    'fixation': 200,    # 10 steps @ dt=20
    'stimulus': 1000,    # 50 steps (rounded to 38 at dt=20)
    'delay':    0,      # 0 steps  — MUST fix stochastic delay
    'decision': 100,    # 5  steps
}
# Total ≈ 58 steps → seq_len buffer = 70

PALETTE = {
    'synergy':    '#54A24B',
    'redundancy': '#4C78A8',
    'unique1':    '#F58518',
    'unique2':    '#E45756',
    'mi_joint':   '#1A1A1A',
    'stim_end':   '#CC0000',
    'context0':   '#7B2D8B',
    'context1':   '#2D8B7B',
}

# ──────────────────────────────────────────────────────────────────────────────
# Fallback CTRNN
# ──────────────────────────────────────────────────────────────────────────────

class _FallbackCTRNN(nn.Module):
    """Euler-discretised CT-RNN: h(t+dt) = h + alpha*(-h + tanh(W_in u + W_rec h))"""
    def __init__(self, input_size, hidden_size, output_size, dt=20, tau=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha  = dt / tau
        self.W_in   = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_rec  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_out  = nn.Linear(hidden_size, output_size, bias=True)
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
# Fallback Gaussian analytic PID  (MMI-PID, Barrett 2015)
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_gaussian_pid(activations, target, timestep=None,
                            n_bipartitions=50, seed=DEFAULT_SEED,
                            regularization=1e-5, **kwargs):
    """
    activations : (N, T, H)
    target      : (N,)  continuous scalar
    Returns dict with synergy / redundancy / unique1 / unique2 / mi_joint
    each of shape (T,) if timestep is None, else scalar.
    """
    rng = np.random.default_rng(seed)
    N, T, H = activations.shape
    y = target.astype(np.float64)

    def _mi(X):
        """Gaussian MI between multivariate X (N,d) and scalar y (N,) via R²."""
        X_ = X - X.mean(0)
        y_ = y - y.mean()
        Sxx = X_.T @ X_ / N + regularization * np.eye(X_.shape[1])
        Sxy = X_.T @ y_ / N
        var_y = np.var(y_) + 1e-12
        try:
            r2 = float(Sxy @ np.linalg.solve(Sxx, Sxy)) / var_y
            r2 = np.clip(r2, 0.0, 1.0 - 1e-9)
        except np.linalg.LinAlgError:
            return 0.0
        return float(-0.5 * np.log(1.0 - r2) / np.log(2))

    def _pid_at_t(h_t):
        syn_l, red_l, u1_l, u2_l, mi_l = [], [], [], [], []
        half = max(H // 2, 1)
        for _ in range(n_bipartitions):
            idx = rng.permutation(H)
            i1, i2 = idx[:half], idx[half:half * 2]
            mi1  = _mi(h_t[:, i1])
            mi2  = _mi(h_t[:, i2])
            mi_j = _mi(h_t[:, np.concatenate([i1, i2])])
            red  = min(mi1, mi2)
            syn  = max(mi_j - mi1 - mi2 + red, 0.0)
            u1   = max(mi1 - red, 0.0)
            u2   = max(mi2 - red, 0.0)
            syn_l.append(syn); red_l.append(red)
            u1_l.append(u1);   u2_l.append(u2)
            mi_l.append(mi_j)
        return {k: float(np.mean(v))
                for k, v in zip(['syn','red','u1','u2','mi'],
                                [syn_l,red_l,u1_l,u2_l,mi_l])}

    def _wrap(r):
        return {'synergy': r['syn'], 'redundancy': r['red'],
                'unique1': r['u1'], 'unique2': r['u2'], 'mi_joint': r['mi']}

    if timestep is not None:
        return _wrap(_pid_at_t(activations[:, timestep].astype(np.float64)))

    results = {k: [] for k in ['syn','red','u1','u2','mi']}
    for t in range(T):
        r = _pid_at_t(activations[:, t].astype(np.float64))
        for k in results:
            results[k].append(r[k])
    return {'synergy':    np.array(results['syn']),
            'redundancy': np.array(results['red']),
            'unique1':    np.array(results['u1']),
            'unique2':    np.array(results['u2']),
            'mi_joint':   np.array(results['mi'])}


# ──────────────────────────────────────────────────────────────────────────────
# Fallback dataset generator
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_dataset_generator(task, dt=DT, batch_size=DEFAULT_BATCH_SIZE,
                                 seq_len=120):
    env = ngym.make(task, dt=dt, timing={'delay': 0}, use_expl_context=True)
    env.reset(seed=DEFAULT_SEED)
    print(type(env))
    print(env.observation_space.shape)

    def _gen():
        obs_b, gt_b = [], []
        for _ in range(batch_size):
            env.unwrapped.new_trial()
            ob = env.unwrapped.ob
            gt = env.unwrapped.gt
            T  = ob.shape[0]
            t  = min(T, seq_len)
            ob_p = np.zeros((seq_len, ob.shape[1]), dtype=np.float32)
            gt_p = np.zeros((seq_len,), dtype=np.int64)
            ob_p[:t] = ob[:t]
            gt_p[:t] = gt[:t]
            if t < seq_len:
                ob_p[t:] = ob[-1]
                gt_p[t:] = gt[-1]
            obs_b.append(ob_p)
            gt_b.append(gt_p)
        return np.stack(obs_b, axis=1), np.stack(gt_b, axis=1)  # (T,B,F),(T,B)

    return _gen


# ──────────────────────────────────────────────────────────────────────────────
# Select implementations
# ──────────────────────────────────────────────────────────────────────────────

CTRNNImpl    = CTRNN                    if _USING_PROJECT_MODULES else _FallbackCTRNN
pid_fn       = gaussian_pid_rnn         if _USING_PROJECT_MODULES else _fallback_gaussian_pid
make_dataset = create_dataset_generator if _USING_PROJECT_MODULES else _fallback_dataset_generator


# ──────────────────────────────────────────────────────────────────────────────
# Wrapped CTRNN — uniform (outputs, hidden_states) interface
# ──────────────────────────────────────────────────────────────────────────────

class WrappedCTRNN(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        if _USING_PROJECT_MODULES:
            self.model = CTRNNImpl(input_size=input_dim,
                                   hidden_size=hidden_size,
                                   output_size=OUTPUT_DIM)
        else:
            self.model = CTRNNImpl(input_size=input_dim,
                                   hidden_size=hidden_size,
                                   output_size=OUTPUT_DIM, dt=DT)

    def forward(self, x):
        if _USING_PROJECT_MODULES:
            outputs, _, hidden_states = self.model(x, return_dynamics=True)
        else:
            outputs, hidden_states = self.model(x)
        return outputs, hidden_states


# ──────────────────────────────────────────────────────────────────────────────
# Loss — only penalise decision timesteps
# ──────────────────────────────────────────────────────────────────────────────

def masked_cross_entropy(outputs, targets):
    mask = targets != 0
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=outputs.device)
    return F.cross_entropy(outputs[mask], targets[mask])


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_ctrnn(hidden_size, n_batches, device, weights_dir,
                n_epochs=5, seed=DEFAULT_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = WrappedCTRNN(INPUT_DIM, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LR)

    if _USING_PROJECT_MODULES:
        dataset = make_dataset(TASK)
    else:
        dataset = make_dataset(TASK, dt=DT)

    print(f"\n  Training CTRNN  hidden={hidden_size}  epochs={n_epochs}  batches/epoch={n_batches}")
    losses = []

    for epoch in range(n_epochs):
        for i in range(n_batches):
            inputs_np, targets_np = dataset()
            inputs  = torch.from_numpy(inputs_np).float().transpose(0, 1).to(device)
            targets = torch.from_numpy(targets_np).long().transpose(0, 1).to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = masked_cross_entropy(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            batch_num = epoch * n_batches + i + 1
            if (i + 1) % 200 == 0 or (i + 1) == n_batches:
                print(f"    epoch {epoch+1}/{n_epochs} batch {i+1:>4}/{n_batches}  "
                      f"loss={np.mean(losses[-50:]):.4f}")

    save_path = os.path.join(weights_dir, f'ctrnn_cdm_h{hidden_size}.pt')
    torch.save(model.state_dict(), save_path)
    print(f"    Saved → {save_path}")
    return model, losses


# ──────────────────────────────────────────────────────────────────────────────
# Test batch — fixed timing, collect rich metadata
# ──────────────────────────────────────────────────────────────────────────────

def get_test_batch(n_trials, device, seed=DEFAULT_SEED):
    """
    Returns
    -------
    inputs          : (B, T, 7)  tensor on device
    targets         : (B, T)     tensor on device
    signed_coh_rel  : (B,)  np  signed coherence of the RELEVANT modality
    context_labels  : (B,)  np  int, 0 or 1
    """
    env = ngym.make(TASK, dt=DT, timing=EVAL_TIMING, use_expl_context=True)
    env.reset(seed=seed)

    obs_l, gt_l, coh_l, ctx_l = [], [], [], []

    for _ in range(n_trials):
        env.unwrapped.new_trial()
        ob    = env.unwrapped.ob.astype(np.float32)   # (T, 7)
        gt    = env.unwrapped.gt.astype(np.int64)      # (T,)
        trial = env.unwrapped.trial

        context      = int(trial['context'])            # 0 or 1
        ground_truth = int(trial['ground_truth'])       # 1 or 2
        coh_0        = float(trial['coh_1'])
        coh_1        = float(trial['coh_2'])

        # Signed coherence of the RELEVANT modality:
        # context=0 → attend modality 0 → relevant coh = coh_0
        # context=1 → attend modality 1 → relevant coh = coh_1
        # Sign: +coh if ground_truth==1, -coh if ground_truth==2
        rel_coh = coh_0 if context == 0 else coh_1
        sign    = 1 if ground_truth == 1 else -1
        signed_coh = sign * rel_coh

        obs_l.append(ob)
        gt_l.append(gt)
        coh_l.append(signed_coh)
        ctx_l.append(context)

    inputs  = torch.tensor(np.stack(obs_l), dtype=torch.float32).to(device)
    targets = torch.tensor(np.stack(gt_l),  dtype=torch.long).to(device)
    cohs    = np.array(coh_l, dtype=np.float32)
    ctxs    = np.array(ctx_l, dtype=np.int32)

    return inputs, targets, cohs, ctxs


# ──────────────────────────────────────────────────────────────────────────────
# Stim-end detection (shared helper)
# ──────────────────────────────────────────────────────────────────────────────

def find_stim_end(inputs_np):
    """
    inputs_np : (B, T, 7)
    Returns median timestep of last stimulus step across trials.
    Stimulus period = fixation channel (ch0) is 0.
    """
    fix_ch    = inputs_np[:, :, 0]  # (B, T)
    stim_ends = np.array([
        np.where(fix_ch[b] < 0.5)[0][-1]
        if (fix_ch[b] < 0.5).any() else 0
        for b in range(fix_ch.shape[0])
    ])
    return int(np.median(stim_ends))


# ──────────────────────────────────────────────────────────────────────────────
# PID analysis — run for two targets
# ──────────────────────────────────────────────────────────────────────────────

def run_pid(model, inputs, targets, cohs, ctxs, n_bipartitions, device):
    """
    Runs time-resolved Gaussian PID with two different targets:
      'coherence' — signed coherence of the relevant modality
      'context'   — which rule is active (0 or 1)

    Returns dict with keys 'coherence' and 'context', each containing
    the standard PID atom arrays plus stim_end and seq_len.
    """
    model.eval()
    with torch.no_grad():
        _, h_seq = model(inputs)               # (B, T, H)
    acts       = h_seq.cpu().numpy()           # (B, T, H)
    inputs_np  = inputs.cpu().numpy()
    stim_end   = find_stim_end(inputs_np)

    results = {}

    for target_name, target_vals in [('coherence', cohs.astype(float)),
                                      ('context',   ctxs.astype(float))]:
        print(f"    PID target: {target_name}  "
              f"| H={acts.shape[2]}  T={acts.shape[1]}")

        pid_out = pid_fn(
            activations=acts,
            target=target_vals,
            timestep=None,
            n_bipartitions=n_bipartitions,
            seed=DEFAULT_SEED,
            regularization=1e-5,
        )

        results[target_name] = {
            'synergy':    np.array(pid_out['synergy']),
            'redundancy': np.array(pid_out['redundancy']),
            'unique1':    np.array(pid_out['unique1']),
            'unique2':    np.array(pid_out['unique2']),
            'mi_joint':   np.array(pid_out['mi_joint']),
            'stim_end':   stim_end,
            'seq_len':    acts.shape[1],
        }

        tend = stim_end
        print(f"      Stim-end t={tend} | "
              f"Syn={results[target_name]['synergy'][tend]:.4f}b | "
              f"Red={results[target_name]['redundancy'][tend]:.4f}b")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Plot — per hidden size (loss + two PID panels)
# ──────────────────────────────────────────────────────────────────────────────

def plot_timeseries(pid_results, hidden_size, losses, save_path):
    """
    Three-panel figure:
      Left  : training loss curve
      Middle: PID with coherence as target
      Right : PID with context as target
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             gridspec_kw={'width_ratios': [1, 2, 2]})
    fig.suptitle(
        f'CTRNN  hidden={hidden_size}  —  ContextDecisionMaking-v0',
        fontsize=13, fontweight='bold', y=1.01
    )

    # ── Loss ──
    ax = axes[0]
    smooth = np.convolve(losses, np.ones(20) / 20, mode='valid')
    ax.plot(losses, alpha=0.2, color='steelblue', lw=0.8)
    ax.plot(np.arange(19, len(losses)), smooth,
            color='steelblue', lw=2, label='Loss (20-batch avg)')
    ax.set_xlabel('Training batch')
    ax.set_ylabel('Masked CE loss')
    ax.set_title('Training loss')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── PID panels ──
    panel_cfg = [
        ('coherence', 'PID — Target: signed coherence\n(relevant modality)'),
        ('context',   'PID — Target: context label\n(which rule is active)'),
    ]

    for ax, (tgt_name, title) in zip(axes[1:], panel_cfg):
        pid  = pid_results[tgt_name]
        T    = pid['seq_len']
        t    = np.arange(T)
        tend = pid['stim_end']

        ax.plot(t, pid['mi_joint'],   label='Total MI',   color=PALETTE['mi_joint'],   lw=2.5, ls=':')
        ax.plot(t, pid['synergy'],    label='Synergy',    color=PALETTE['synergy'],    lw=2.5)
        ax.plot(t, pid['redundancy'], label='Redundancy', color=PALETTE['redundancy'], lw=2.5)
        ax.plot(t, pid['unique1'],    label='Unique 1',   color=PALETTE['unique1'],    lw=1.5, alpha=0.8)
        ax.plot(t, pid['unique2'],    label='Unique 2',   color=PALETTE['unique2'],    lw=1.5, alpha=0.8)

        ax.axvline(tend, color=PALETTE['stim_end'], ls='--', lw=1.5,
                   label=f'Stim end (t={tend})')
        ax.axvspan(0, tend, color='gray', alpha=0.08)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f'Timestep (dt={DT}ms)')
        ax.set_ylabel('Information (bits)')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot — comparison summary across all hidden sizes
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison(all_results, save_path):
    hidden_sizes = sorted(all_results.keys(), reverse=True)
    n    = len(hidden_sizes)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n))

    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)

    # Row 0: synergy over time — coherence target (left) and context target (right)
    ax_syn_coh = fig.add_subplot(gs[0, :2])
    ax_syn_ctx = fig.add_subplot(gs[0, 2])

    # Row 1: redundancy over time — coherence (left) and context (right)
    ax_red_coh = fig.add_subplot(gs[1, :2])
    ax_red_ctx = fig.add_subplot(gs[1, 2])

    # Row 2: bar charts at stim-end
    ax_bar_coh = fig.add_subplot(gs[2, 0])
    ax_bar_ctx = fig.add_subplot(gs[2, 1])
    ax_ratio   = fig.add_subplot(gs[2, 2])

    # Row 3: scalar summaries
    ax_loss    = fig.add_subplot(gs[3, 0])
    ax_syn_end = fig.add_subplot(gs[3, 1])
    ax_red_end = fig.add_subplot(gs[3, 2])

    fig.suptitle(
        'CTRNN PID Sweep — ContextDecisionMaking-v0\n'
        'Left/Middle: coherence target  |  Right: context target',
        fontsize=13, fontweight='bold'
    )

    stim_end_ref = all_results[hidden_sizes[0]]['coherence']['stim_end']

    # ── Synergy over time ──
    for panel, tgt in [(ax_syn_coh, 'coherence'), (ax_syn_ctx, 'context')]:
        for i, h in enumerate(hidden_sizes):
            pid = all_results[h][tgt]
            panel.plot(np.arange(pid['seq_len']), pid['synergy'],
                       color=cmap[i], lw=2, label=f'H={h}')
        panel.axvline(stim_end_ref, color='red', ls='--', lw=1, alpha=0.6)
        panel.set_title(f'Synergy over time — target: {tgt}')
        panel.set_ylabel('Synergy (bits)')
        panel.set_xlabel(f'Timestep (dt={DT}ms)')
        panel.legend(ncol=n, fontsize=7, loc='upper left')
        panel.grid(alpha=0.25)

    # ── Redundancy over time ──
    for panel, tgt in [(ax_red_coh, 'coherence'), (ax_red_ctx, 'context')]:
        for i, h in enumerate(hidden_sizes):
            pid = all_results[h][tgt]
            panel.plot(np.arange(pid['seq_len']), pid['redundancy'],
                       color=cmap[i], lw=2, label=f'H={h}')
        panel.axvline(stim_end_ref, color='red', ls='--', lw=1, alpha=0.6)
        panel.set_title(f'Redundancy over time — target: {tgt}')
        panel.set_ylabel('Redundancy (bits)')
        panel.set_xlabel(f'Timestep (dt={DT}ms)')
        panel.legend(ncol=n, fontsize=7, loc='upper left')
        panel.grid(alpha=0.25)

    # ── Bar charts at stim-end ──
    atoms       = ['synergy', 'redundancy', 'unique1', 'unique2']
    atom_colors = [PALETTE['synergy'], PALETTE['redundancy'],
                   PALETTE['unique1'], PALETTE['unique2']]
    x      = np.arange(len(hidden_sizes))
    width  = 0.18
    offs   = np.linspace(-0.27, 0.27, 4)

    for ax_bar, tgt in [(ax_bar_coh, 'coherence'), (ax_bar_ctx, 'context')]:
        for j, (atom, color) in enumerate(zip(atoms, atom_colors)):
            vals = [all_results[h][tgt][atom][all_results[h][tgt]['stim_end']]
                    for h in hidden_sizes]
            ax_bar.bar(x + offs[j], vals, width,
                       label=atom.capitalize(), color=color, alpha=0.85)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f'H={h}' for h in hidden_sizes], fontsize=8)
        ax_bar.set_title(f'PID atoms at stim-end — target: {tgt}')
        ax_bar.set_ylabel('Information (bits)')
        ax_bar.legend(fontsize=7)
        ax_bar.grid(axis='y', alpha=0.3)

    # ── Synergy ratio: Syn/(Syn+Red) at stim-end for both targets ──
    x2 = np.arange(len(hidden_sizes))
    w2 = 0.35
    for j, (tgt, color) in enumerate([('coherence', PALETTE['synergy']),
                                       ('context',   PALETTE['context0'])]):
        ratios = []
        for h in hidden_sizes:
            pid  = all_results[h][tgt]
            tend = pid['stim_end']
            syn  = pid['synergy'][tend]
            red  = pid['redundancy'][tend]
            denom = syn + red
            ratios.append(syn / denom if denom > 1e-10 else 0.0)
        ax_ratio.bar(x2 + (j - 0.5) * w2, ratios, w2,
                     label=f'target: {tgt}', color=color, alpha=0.8)
    ax_ratio.axhline(0.5, color='gray', ls='--', lw=1)
    ax_ratio.set_xticks(x2)
    ax_ratio.set_xticklabels([f'H={h}' for h in hidden_sizes], fontsize=8)
    ax_ratio.set_ylim(0, 1)
    ax_ratio.set_title('Synergy ratio at stim-end\n[Syn / (Syn+Red)]')
    ax_ratio.set_ylabel('Synergy ratio')
    ax_ratio.legend(fontsize=7)
    ax_ratio.grid(axis='y', alpha=0.3)

    # ── Final loss vs hidden size ──
    final_losses = [np.mean(all_results[h]['losses'][-50:])
                    for h in hidden_sizes]
    ax_loss.plot([str(h) for h in hidden_sizes], final_losses,
                 'o-', color='steelblue', lw=2, ms=8)
    ax_loss.set_title('Final training loss vs hidden size')
    ax_loss.set_xlabel('Hidden size')
    ax_loss.set_ylabel('Loss (last 50 batches avg)')
    ax_loss.grid(alpha=0.3)

    # ── Synergy at stim-end vs hidden size (both targets) ──
    for tgt, color, marker in [('coherence', PALETTE['synergy'],    'o'),
                                ('context',   PALETTE['context0'],   's')]:
        vals = [all_results[h][tgt]['synergy'][all_results[h][tgt]['stim_end']]
                for h in hidden_sizes]
        ax_syn_end.plot([str(h) for h in hidden_sizes], vals,
                        f'{marker}-', color=color, lw=2, ms=8, label=f'target: {tgt}')
    ax_syn_end.set_title('Synergy at stim-end vs hidden size')
    ax_syn_end.set_xlabel('Hidden size')
    ax_syn_end.set_ylabel('Synergy (bits)')
    ax_syn_end.legend(fontsize=8)
    ax_syn_end.grid(alpha=0.3)

    # ── Redundancy at stim-end vs hidden size (both targets) ──
    for tgt, color, marker in [('coherence', PALETTE['redundancy'],  'o'),
                                ('context',   PALETTE['context1'],    's')]:
        vals = [all_results[h][tgt]['redundancy'][all_results[h][tgt]['stim_end']]
                for h in hidden_sizes]
        ax_red_end.plot([str(h) for h in hidden_sizes], vals,
                        f'{marker}-', color=color, lw=2, ms=8, label=f'target: {tgt}')
    ax_red_end.set_title('Redundancy at stim-end vs hidden size')
    ax_red_end.set_xlabel('Hidden size')
    ax_red_end.set_ylabel('Redundancy (bits)')
    ax_red_end.legend(fontsize=8)
    ax_red_end.grid(alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison figure saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(hidden_sizes, n_train_batches, n_test_trials):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device      : {device}")
    print(f"Task        : {TASK}")
    print(f"Input dim   : {INPUT_DIM}  (fixation + 4 stim channels + 2 context channels)")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Train batches per model: {n_train_batches}")
    print(f"Test trials for PID    : {n_test_trials}")
    print(f"Eval timing (fixed)    : {EVAL_TIMING}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    weights_dir = os.path.join(RESULTS_DIR, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # One shared test batch for all models
    print("\nGenerating fixed test batch...")
    inputs, targets, cohs, ctxs = get_test_batch(n_test_trials, device)
    print(f"  inputs : {tuple(inputs.shape)}")
    print(f"  cohs   : range [{cohs.min():.1f}, {cohs.max():.1f}]  "
          f"std={cohs.std():.2f}")
    print(f"  context: {np.bincount(ctxs)}  (count per context label)")

    all_results = {}

    for h in hidden_sizes:
        print(f"\n{'='*60}")
        print(f"  Hidden size = {h}")
        print(f"{'='*60}")

        model, losses = train_ctrnn(h, n_train_batches, device, weights_dir, n_epochs=args.n_epochs)

        n_bipartitions = 2 * h
        print(f"  Running PID (n_bipartitions={n_bipartitions})...")
        pid_data = run_pid(model, inputs, targets, cohs, ctxs,
                           n_bipartitions, device)

        all_results[h] = {**pid_data, 'losses': losses}

        ts_path = os.path.join(RESULTS_DIR, f'pid_timeseries_h{h}.png')
        plot_timeseries(pid_data, h, losses, ts_path)

    # Comparison summary
    comp_path = os.path.join(RESULTS_DIR, 'comparison_summary.png')
    plot_comparison(all_results, comp_path)

    # Save raw arrays
    npz_path = os.path.join(RESULTS_DIR, 'sweep_results.npz')
    save_dict = {}
    for h, r in all_results.items():
        for tgt in ('coherence', 'context'):
            for atom in ('synergy', 'redundancy', 'unique1', 'unique2', 'mi_joint'):
                save_dict[f'h{h}_{tgt}_{atom}'] = r[tgt][atom]
            save_dict[f'h{h}_{tgt}_stim_end'] = np.array(r[tgt]['stim_end'])
        save_dict[f'h{h}_losses'] = np.array(r['losses'])
    np.savez_compressed(npz_path, **save_dict)
    print(f"Raw results → {npz_path}")
    print("\nDone.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='CTRNN PID sweep — ContextDecisionMaking-v0')
    p.add_argument('--hidden_sizes',    nargs='+', type=int,
                   default=DEFAULT_HIDDEN_SIZES)
    p.add_argument('--n_train_batches', type=int, default=DEFAULT_N_TRAIN)
    p.add_argument('--n_test_trials',   type=int, default=DEFAULT_N_TEST)
    p.add_argument('--n_epochs', type=int, default=1)
    args = p.parse_args()

    main(
        hidden_sizes    = args.hidden_sizes,
        n_train_batches = args.n_train_batches,
        n_test_trials   = args.n_test_trials,
    )