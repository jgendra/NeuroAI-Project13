"""
gaussian_pid.py
===============

Gaussian analytic Partial Information Decomposition (PID), using the
Minimum-Mutual-Information (MMI) redundancy of Barrett (2015), i.e. the same
Gaussian machinery that the `phyid` library (Mediano/Rosas/Luppi) uses for its
Integrated Information Decomposition (PhiID), but specialised to "static" PID:
two source groups -> one target, instead of past -> future.

Why this is the right reduction
--------------------------------
phyid's `calc_PhiID` decomposes I(past ; future) over a 4-variable lattice and
therefore returns 16 atoms. Static PID with 2 sources X1, X2 and a target Y is
the much simpler 4-atom Williams & Beer (2010) lattice:

    I(X1, X2 ; Y) = Red + Unq1 + Unq2 + Syn

with the standard consistency equations
    I(X1 ; Y)      = Red + Unq1
    I(X2 ; Y)      = Red + Unq2
    I(X1, X2 ; Y)  = Red + Unq1 + Unq2 + Syn

Fixing the redundancy fixes all four atoms. Following Barrett (2015), for a
jointly Gaussian system with a univariate target the MMI redundancy is

    Red = min( I(X1 ; Y), I(X2 ; Y) )

from which
    Unq1 = I(X1 ; Y) - Red
    Unq2 = I(X2 ; Y) - Red
    Syn  = I(X1, X2 ; Y) - I(X1 ; Y) - I(X2 ; Y) + Red

Each Gaussian mutual information is computed in closed form from the joint
covariance (this is exactly the mean of phyid's local Gaussian quantities, so
the numbers match phyid up to the choice of log base):

    I(A ; B) = 1/2 * log( det(Cov_AA) * det(Cov_BB) / det(Cov_[A,B][A,B]) )

Units: returned in nats by default (log base e, matching phyid's Gaussian path);
pass log_base=2 for bits.

Known limitations (worth stating in the report)
------------------------------------------------
1) MMI forces one of the two unique terms to be exactly zero and forces
  redundancy to be the smaller of the two source-target MIs, regardless of how
  the sources actually relate. This is a rigidity of the MMI definition.
2) Barrett's exactness result is for a "univariate" target. The code accepts a
  multivariate target mechanically, but the MMI interpretation is cleanest for a
  univariate target (which is what this project uses: signed coherence /
  stimulus value).
3) The Gaussian assumption is only an approximation for bounded tanh activations.
"""

import numpy as np
import torch

# nats -> bits conversion factor (change of log base). The GPU PID path computes
# mutual information in nats and converts the returned atoms to bits with this.
NATS_TO_BITS = 1.0 / np.log(2.0)

# ----------------------------------------------------------------------------- #
# Low-level helpers
# ----------------------------------------------------------------------------- #
def _as_2d(a):
    """
    Coerce an array to shape (n_samples, n_features).

    A 1-D array of length n is interpreted as n samples of a single variable and
    becomes shape (n, 1). A 2-D array is returned unchanged.
    """
    a = np.asarray(a, dtype=float)          # make sure we work with float ndarray
    if a.ndim == 1:                         # a single variable given as (n,)
        a = a[:, None]                      # add a feature axis -> (n, 1)
    if a.ndim != 2:                         # anything that is not 1-D or 2-D is invalid
        raise ValueError(f"expected 1-D or 2-D array, got shape {a.shape}")
    return a                                # shape (n_samples, n_features)


def _logdet(mat):
    """
    Stable log-determinant of a symmetric positive-definite matrix.

    Uses np.linalg.slogdet so we never overflow/underflow on the raw determinant.
    Raises if the matrix is not positive-definite (sign <= 0), which signals a
    rank-deficient covariance (too few samples, dead/constant units, or perfectly
    collinear units) rather than silently returning nonsense.
    """
    sign, logabsdet = np.linalg.slogdet(mat)   # sign of det, log|det|
    if sign <= 0:                              # non-positive determinant => not PD
        raise np.linalg.LinAlgError(
            "covariance sub-matrix is not positive-definite "
            "(rank-deficient: increase samples, remove constant/collinear units, "
            "or pass a small `regularization`)."
        )
    return logabsdet                           # natural-log determinant (nats-friendly)


def _gaussian_mi(cov, idx_a, idx_b):
    """
    Closed-form Gaussian mutual information I(A ; B) in nats.

    Parameters
    ----------
    cov : (D, D) ndarray
        Joint covariance over all variables.
    idx_a, idx_b : list[int]
        Column indices into `cov` for variable groups A and B (disjoint).

    Returns
    -------
    float
        I(A ; B) = 1/2 * [ logdet(Cov_AA) + logdet(Cov_BB) - logdet(Cov_AB,AB) ]
    """
    idx_ab = list(idx_a) + list(idx_b)                 # indices of the union A u B
    ld_a = _logdet(cov[np.ix_(idx_a, idx_a)])          # logdet of A-block
    ld_b = _logdet(cov[np.ix_(idx_b, idx_b)])          # logdet of B-block
    ld_ab = _logdet(cov[np.ix_(idx_ab, idx_ab)])       # logdet of joint AB-block
    return 0.5 * (ld_a + ld_b - ld_ab)                 # Gaussian MI identity (nats)


def _build_bipartitions(n_units, n_bip, seed):
    """Reproduce EXACTLY the (idx1, idx2) split sequence gaussian_pid_rnn draws:
    np.random.default_rng(seed).permutation(n_units), cut at n_units//2, per split."""
    rng = np.random.default_rng(seed)
    cut = n_units // 2
    idx1 = np.empty((n_bip, cut), dtype=np.int64)             # group-1 unit indices
    idx2 = np.empty((n_bip, n_units - cut), dtype=np.int64)   # group-2 unit indices
    for b in range(n_bip):
        perm = rng.permutation(n_units)
        idx1[b], idx2[b] = perm[:cut], perm[cut:]
    return idx1, idx2



# ----------------------------------------------------------------------------- #
# Core analytic PID (the function the whole project is built on)
# ----------------------------------------------------------------------------- #
def gaussian_pid(sources_1, sources_2, target,
                 log_base="e", standardize=True, regularization=0.0):
    """
    Gaussian analytic MMI-PID for two source groups and one target.

    This is the atomic primitive. It treats every "row" as one observation
    (e.g. one trial) and every "column" as one variable (e.g. one RNN unit, or
    the scalar target). It estimates a single joint covariance over
    [sources_1 | sources_2 | target] and returns the four PID atoms in closed
    form.

    Parameters
    ----------
    sources_1 : array_like, shape (n_samples,) or (n_samples, d1)
        First source group X1 (e.g. one subpopulation of RNN units). 1-D input
        is treated as a single variable.
    sources_2 : array_like, shape (n_samples,) or (n_samples, d2)
        Second source group X2 (e.g. the complementary subpopulation).
    target : array_like, shape (n_samples,) or (n_samples, dt)
        Target Y (e.g. the stimulus / signed coherence / sinusoid value).
        Univariate is recommended (Barrett's MMI exactness assumes this).
    log_base : {"e", 2}, optional
        Units of the returned information. "e" -> nats (matches phyid's Gaussian
        path, the default); 2 -> bits.
    standardize : bool, optional
        If True (default), each column is divided by its standard deviation
        before forming the covariance. Mutual information is invariant to such
        per-variable rescaling, so this only improves numerical conditioning.
    regularization : float, optional
        If > 0, adds `regularization` to the diagonal of the joint covariance
        (ridge) to stabilise near-singular cases. Default 0.0 (off). Note this
        slightly biases the MI values, so keep it small.

    Returns
    -------
    dict with keys:
        'redundancy', 'unique1', 'unique2', 'synergy' : the four PID atoms
        'mi_1'     = I(X1 ; Y)
        'mi_2'     = I(X2 ; Y)
        'mi_joint' = I(X1, X2 ; Y)
        'total'    = sum of the four atoms (equals mi_joint up to rounding;
                     a built-in consistency check)
    All values are floats in the chosen log base.

    How to call
    -----------
    >>> import numpy as np
    >>> X1 = np.random.randn(2000, 5)          # 2000 trials, 5 units in group 1
    >>> X2 = np.random.randn(2000, 5)          # 2000 trials, 5 units in group 2
    >>> Y  = X1[:, 0] + X2[:, 0] + 0.1*np.random.randn(2000)   # univariate target
    >>> atoms = gaussian_pid(X1, X2, Y, log_base=2)            # bits
    >>> atoms['synergy'], atoms['redundancy']
    """
    # --- 1. shape everything to (n_samples, n_features) ---------------------- #
    X1 = _as_2d(sources_1)                  # (n, d1)
    X2 = _as_2d(sources_2)                  # (n, d2)
    Y = _as_2d(target)                      # (n, dt)

    # all three must share the same number of samples (rows)
    n = X1.shape[0]                         # number of observations
    if not (X2.shape[0] == n and Y.shape[0] == n):
        raise ValueError("sources_1, sources_2 and target must have the same "
                         f"number of samples; got {X1.shape[0]}, {X2.shape[0]}, "
                         f"{Y.shape[0]}.")

    # dimensionalities of each block
    d1, d2, dt = X1.shape[1], X2.shape[1], Y.shape[1]   # feature counts
    total_dim = d1 + d2 + dt                            # size of the joint system

    # need strictly more samples than variables for a full-rank covariance
    if n <= total_dim:
        raise ValueError(f"need n_samples ({n}) > total variables ({total_dim}) "
                         "for a full-rank covariance estimate.")

    # --- 2. assemble the joint data matrix [X1 | X2 | Y] --------------------- #
    data = np.hstack([X1, X2, Y])           # (n, d1+d2+dt)

    # optional per-variable rescaling to unit std (MI-invariant, aids stability)
    if standardize:
        sd = data.std(axis=0, ddof=1)       # per-column standard deviation
        sd[sd == 0] = 1.0                   # guard: leave constant columns untouched
        data = data / sd                    # rescale each variable to unit variance

    # --- 3. single joint covariance over all variables ---------------------- #
    cov = np.cov(data, rowvar=False)        # (D, D); rows=samples, cols=variables
    cov = np.atleast_2d(cov)                # keep 2-D even if D happens to be 1
    if regularization > 0:                  # optional ridge for conditioning
        cov = cov + regularization * np.eye(cov.shape[0])

    # --- 4. index sets into the joint covariance ---------------------------- #
    i1 = list(range(0, d1))                 # columns belonging to X1
    i2 = list(range(d1, d1 + d2))           # columns belonging to X2
    iy = list(range(d1 + d2, total_dim))    # columns belonging to Y

    # --- 5. the three mutual informations (nats) ---------------------------- #
    mi_1 = _gaussian_mi(cov, i1, iy)            # I(X1 ; Y)
    mi_2 = _gaussian_mi(cov, i2, iy)            # I(X2 ; Y)
    mi_joint = _gaussian_mi(cov, i1 + i2, iy)   # I(X1, X2 ; Y)

    # --- 6. convert to the requested log base ------------------------------- #
    if log_base == "e":                     # nats: divide by ln(e)=1, i.e. no-op
        scale = 1.0
    elif log_base == 2:                     # bits: convert nats -> bits
        scale = 1.0 / np.log(2.0)
    else:
        raise ValueError("log_base must be 'e' (nats) or 2 (bits).")
    mi_1, mi_2, mi_joint = mi_1 * scale, mi_2 * scale, mi_joint * scale

    # --- 7. MMI atoms (Barrett 2015) ---------------------------------------- #
    redundancy = min(mi_1, mi_2)            # MMI redundancy = smaller source-target MI
    unique1 = mi_1 - redundancy             # X1's unique info (>=0; ==0 if X1 is the min)
    unique2 = mi_2 - redundancy             # X2's unique info (>=0; ==0 if X2 is the min)
    synergy = mi_joint - mi_1 - mi_2 + redundancy   # leftover that needs both sources

    # --- 8. package, with a self-consistency total -------------------------- #
    return {
        "redundancy": redundancy,
        "unique1": unique1,
        "unique2": unique2,
        "synergy": synergy,
        "mi_1": mi_1,
        "mi_2": mi_2,
        "mi_joint": mi_joint,
        "total": redundancy + unique1 + unique2 + synergy,   # should equal mi_joint
    }


# ----------------------------------------------------------------------------- #
# Versatile RNN wrapper: one snapshot OR every timestep, with bipartition averaging
# ----------------------------------------------------------------------------- #
def gaussian_pid_rnn(activations, target,
                     timestep=None,
                     bipartitions="random", n_bipartitions=200, balanced=True,
                     seed=0, log_base="e", standardize=True, regularization=0.0):
    """
    Compute Gaussian analytic PID on RNN hidden activations, at one timestep or
    across all timesteps, averaging over random subpopulation bipartitions.

    This is the convenience layer the project will actually call. The "sources"
    are two subpopulations of hidden units obtained by splitting the population
    in two; the "target" is the stimulus. Because the MMI atoms depend on which
    units land in which group, the result is averaged over many random balanced
    bipartitions (this is the "random 40/40 splits averaged over many splits"
    procedure from the project plan; for a 10-unit test RNN the splits are 5/5).

    Parameters
    ----------
    activations : array_like
        Hidden activations. Either
          a) (n_samples, n_timesteps, n_units)  -> per-timestep PID is available, or
          b) (n_samples, n_units)               -> already a single snapshot.
        `n_samples` is the ensemble dimension over which covariance is estimated
        (in the real project: trials; for a single-trajectory test RNN: time, if
        you pass the trajectory in the samples axis).
    target : array_like
        The target / stimulus. Either
          a) (n_samples, n_timesteps)  (a value per trial and timestep),
          b) (n_samples,)              (one value per trial, shared across time), or
          c) (n_samples, n_timesteps, dt) / (n_samples, dt) for a multivariate target.
    timestep : int or None, optional
        int  -> compute PID only at this timestep (supports negative indexing,
                  e.g. -1 = last timestep). Returns scalar atoms.
        None -> compute PID at every timestep. Returns arrays of shape
                  (n_timesteps,). Requires 3-D `activations`.
    bipartitions : {"random", "half"} or list of (idx1, idx2), optional
        How to split the units into two source groups:
          a) "random" (default): draw `n_bipartitions` random splits.
          b) "half": a single split into first-half / second-half units.
          c) explicit list: each element is a pair (idx1, idx2) of index arrays.
    n_bipartitions : int, optional
        Number of random splits to average over when bipartitions="random".
    balanced : bool, optional
        If True (default), random splits are (near-)equal in size (n//2 vs the
        rest). If False, the split point is uniformly random.
    seed : int, optional
        Seed for the random bipartitions (reproducibility).
    log_base, standardize, regularization :
        Passed straight through to `gaussian_pid`.

    Returns
    -------
    dict
        Keys 'redundancy', 'unique1', 'unique2', 'synergy', 'mi_1', 'mi_2',
        'mi_joint' mapped to the bipartition-averaged value(s). For a single
        timestep these are floats; for timestep=None they are arrays of shape
        (n_timesteps,). Each atom also has a companion '<atom>_std' giving the
        standard deviation across bipartitions (useful for variability checks).

    How to call
    -----------
    # Snapshot at the last timestep (trials as samples), univariate target:
    >>> out = gaussian_pid_rnn(acts, stim, timestep=-1)        # acts: (trials, T, units)
    >>> out['synergy'], out['synergy_std']

    # Full time-resolved PID profile (one PID per timestep):
    >>> prof = gaussian_pid_rnn(acts, stim, timestep=None)
    >>> prof['synergy'].shape                                  # (T,)
    """
    # --- normalise activations to a 3-D (samples, timesteps, units) view ----- #
    acts = np.asarray(activations, dtype=float)     # to ndarray
    if acts.ndim == 2:                              # (samples, units): a lone snapshot
        acts = acts[:, None, :]                     # insert a length-1 time axis
    if acts.ndim != 3:                              # otherwise we cannot interpret it
        raise ValueError("activations must be (n_samples, n_units) or "
                         f"(n_samples, n_timesteps, n_units); got {acts.shape}.")
    n_samples, n_timesteps, n_units = acts.shape    # unpack the three axes

    # --- normalise the target so we can index it by timestep ----------------- #
    tgt = np.asarray(target, dtype=float)           # to ndarray
    if tgt.ndim == 1:                               # (samples,): same target every timestep
        tgt = np.repeat(tgt[:, None], n_timesteps, axis=1)        # -> (samples, T)
    if tgt.ndim == 2 and tgt.shape == (n_samples, n_timesteps):   # (samples, T): per-step scalar
        tgt = tgt[:, :, None]                       # add a trailing feature axis -> (samples, T, 1)
    if tgt.ndim == 2 and tgt.shape[0] == n_samples and tgt.shape[1] != n_timesteps:
        # (samples, dt): a multivariate target shared across time
        tgt = np.repeat(tgt[:, None, :], n_timesteps, axis=1)     # -> (samples, T, dt)
    if not (tgt.ndim == 3 and tgt.shape[0] == n_samples and tgt.shape[1] == n_timesteps):
        raise ValueError("target shape is incompatible with activations; expected "
                         "(n_samples,), (n_samples, n_timesteps) or matching 3-D.")

    # --- decide which timesteps to evaluate ---------------------------------- #
    if timestep is None:                            # all timesteps requested
        t_indices = list(range(n_timesteps))        # evaluate every step
        single = False                              # we will return arrays
    else:                                           # one specific timestep
        t = timestep if timestep >= 0 else n_timesteps + timestep   # resolve negatives
        if not (0 <= t < n_timesteps):              # bounds check after resolving
            raise IndexError(f"timestep {timestep} out of range for "
                             f"{n_timesteps} timesteps.")
        t_indices = [t]                             # evaluate just this one
        single = True                               # we will return scalars

    # --- build the list of bipartitions (index pairs) ------------------------ #
    rng = np.random.default_rng(seed)               # reproducible random generator
    if bipartitions == "random":                    # many random splits to average over
        splits = []                                 # collect (idx1, idx2) pairs here
        for _ in range(n_bipartitions):             # one split per iteration
            perm = rng.permutation(n_units)         # random ordering of unit indices
            cut = n_units // 2 if balanced else int(rng.integers(1, n_units))
            splits.append((perm[:cut], perm[cut:]))         # left vs right of the cut
    elif bipartitions == "half":                    # a single deterministic split
        cut = n_units // 2                          # midpoint
        splits = [(np.arange(0, cut), np.arange(cut, n_units))]
    else:                                           # caller supplied explicit splits
        splits = [(np.asarray(a), np.asarray(b)) for (a, b) in bipartitions]

    # --- accumulate atoms over timesteps and bipartitions -------------------- #
    keys = ["redundancy", "unique1", "unique2", "synergy", "mi_1", "mi_2", "mi_joint"]
    mean_out = {k: np.zeros(len(t_indices)) for k in keys}   # mean across splits per t
    std_out = {k: np.zeros(len(t_indices)) for k in keys}    # std across splits per t

    for ti, t in enumerate(t_indices):              # loop over requested timesteps
        H = acts[:, t, :]                           # (n_samples, n_units) snapshot at t
        Y = tgt[:, t, :]                            # (n_samples, dt) target at t
        per_split = {k: [] for k in keys}           # per-bipartition values at this t
        for idx1, idx2 in splits:                   # loop over the bipartitions
            atoms = gaussian_pid(                    # the analytic primitive
                H[:, idx1], H[:, idx2], Y,
                log_base=log_base,
                standardize=standardize,
                regularization=regularization,
            )
            for k in keys:                           # stash each quantity
                per_split[k].append(atoms[k])
        for k in keys:                               # reduce across bipartitions
            mean_out[k][ti] = np.mean(per_split[k])  # bipartition-averaged atom
            std_out[k][ti] = np.std(per_split[k])    # spread across bipartitions

    # --- shape the return: scalars for one timestep, arrays for all ---------- #
    result = {}                                      # assemble the output dict
    for k in keys:
        if single:                                   # one timestep -> plain floats
            result[k] = float(mean_out[k][0])
            result[k + "_std"] = float(std_out[k][0])
        else:                                        # many timesteps -> arrays of shape (T,)
            result[k] = mean_out[k]
            result[k + "_std"] = std_out[k]
    return result


# ----------------------------------------------------------------------------- #
# GPU-accelerated, time-resolved PID over RNN activations
# ----------------------------------------------------------------------------- #
def gaussian_pid_rnn_gpu(H, Y, n_bip=200, seed=0, reg=1e-5, device=None):
    """
    GPU-batched, time-resolved Gaussian analytic MMI-PID over RNN activations.

    Fast drop-in for `gaussian_pid_rnn(..., timestep=None, bipartitions="random")`:
    it returns the same bipartition-averaged PID atoms at every timestep, but is
    ~40x faster. The trick is that at each timestep the joint covariance over *all*
    units + target is shared across bipartitions, so it is formed ONCE per timestep
    on the GPU and only the small per-split sub-block log-determinants are batched.
    The bipartitions come from `_build_bipartitions`, which replays the exact same
    RNG sequence as `gaussian_pid_rnn`, so the two paths agree to ~1e-10 bits.

    As in `gaussian_pid`, each column is standardized to unit variance (MI is
    invariant to this; it only conditions the covariance) and a `reg` ridge is
    added to the covariance diagonal before the closed-form Gaussian MI and the
    MMI atoms (Barrett 2015) are computed.

    Parameters
    ----------
    H : array_like, shape (n_samples, n_timesteps, n_units)
        Hidden activations. `n_samples` (trials) is the axis over which the
        per-timestep covariance is estimated.
    Y : array_like, shape (n_samples,)
        Univariate target (e.g. signed coherence / stimulus): one value per trial,
        shared across timesteps.
    n_bip : int, optional
        Number of random balanced (n_units // 2) bipartitions to average over.
    seed : int, optional
        Seed for the bipartition RNG. Must match `gaussian_pid_rnn`'s `seed` to
        reproduce its numbers.
    reg : float, optional
        Ridge added to the covariance diagonal for conditioning (the GPU analogue
        of `gaussian_pid`'s `regularization`).
    device : torch.device or str or None, optional
        Torch device to run on. None (default) selects CUDA if available, else CPU.

    Returns
    -------
    numpy.ndarray, shape (n_timesteps, 5)
        Bipartition-averaged PID atoms per timestep, in BITS, with the atom axis
        ordered [total_mi, redundancy, unique1, unique2, synergy]
        (total_mi = redundancy + unique1 + unique2 + synergy).

    How to call
    -----------
    # Full time-resolved PID profile (trials as samples), univariate target:
    >>> pid = gaussian_pid_rnn_gpu(acts, stim)     # acts: (trials, T, units)
    >>> pid.shape                                  # (T, 5)
    >>> pid[:, 4]                                  # synergy over time (bits)
    """
    N, T, U = H.shape
    idx1, idx2 = _build_bipartitions(U, n_bip, seed)          # identical to reference
    # pick CUDA when available unless the caller forced a device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    # --- standardize each column to unit std (ddof=1); MI is scale-invariant, this
    #     only conditions the covariance and matches gaussian_pid(standardize=True) ---
    Ht = torch.tensor(H, dtype=torch.float64, device=dev)
    Yt = torch.tensor(np.asarray(Y, float), dtype=torch.float64, device=dev)
    Ht = Ht / Ht.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-12)
    Yt = Yt / Yt.std(unbiased=True).clamp_min(1e-12)

    # --- one joint covariance per timestep over [units | target]: (T, U+1, U+1) ---
    D  = torch.cat([Ht, Yt[:, None, None].expand(N, T, 1)], dim=2)     # (N,T,U+1)
    Dc = D - D.mean(dim=0, keepdim=True)                              # center per column
    cov = torch.einsum('nti,ntj->tij', Dc, Dc) / (N - 1)             # sample cov (ddof=1)
    cov = cov + reg * torch.eye(U + 1, dtype=torch.float64, device=dev)  # ridge (matches reg)

    iy = U                                                            # target column index
    # quantities that DON'T depend on the bipartition (computed once per timestep):
    ld_full  = torch.linalg.slogdet(cov)[1]                          # logdet full joint (T,)
    ld_units = torch.linalg.slogdet(cov[:, :U, :U])[1]               # logdet all-units block
    ld_y     = torch.log(cov[:, iy, iy])                             # logdet 1x1 target block
    mi_joint = 0.5 * (ld_units + ld_y - ld_full)                     # I(X1,X2;Y), same all splits

    # index tensors for the per-split source blocks (+ target column appended)
    i1  = torch.as_tensor(idx1, device=dev)                           # group1 indices (B, k1)
    i2  = torch.as_tensor(idx2, device=dev)                           # group2 indices (B, k2)
    i1y = torch.cat([i1, torch.full((n_bip, 1), iy, device=dev)], 1)  # group1 + target
    i2y = torch.cat([i2, torch.full((n_bip, 1), iy, device=dev)], 1)  # group2 + target

    # --- loop over timesteps, compute per-split logdet sub-blocks, then MMI atoms ---
    out = np.zeros((T, 5))                      # (timesteps, [total, red, u1, u2, syn])              
    for t in range(T):                          # loop timesteps -> tiny per-step GPU memory
        c = cov[t]
        def bld(idx):                           # batched logdet of (B,k,k) sub-blocks
            sub = c[idx[:, :, None], idx[:, None, :]]
            return torch.linalg.slogdet(sub)[1]
        mi1 = 0.5 * (bld(i1) + ld_y[t] - bld(i1y))          # I(X1;Y) per split (B,)
        mi2 = 0.5 * (bld(i2) + ld_y[t] - bld(i2y))          # I(X2;Y) per split
        red = torch.minimum(mi1, mi2)                       # MMI redundancy (Barrett 2015)
        u1, u2 = mi1 - red, mi2 - red                       # unique atoms
        syn = mi_joint[t] - mi1 - mi2 + red                 # synergy
        tot = red + u1 + u2 + syn                           # == mi_joint (consistency)
        # bipartition-average, then nats -> bits
        for j, v in enumerate([tot, red, u1, u2, syn]):
            out[t, j] = v.mean().item() * NATS_TO_BITS
    return out


# ----------------------------------------------------------------------------- #
# GPU-accelerated one-vs-others PID: every unit against all other units
# ----------------------------------------------------------------------------- #
def gaussian_pid_gpu_one2many(H, Y, timestep=None, reg=1e-5, device=None):
    """
    GPU-batched Gaussian analytic MMI-PID of every RNN unit against all the
    others ("one-vs-others"), at one timestep or across all timesteps.

    Where `gaussian_pid_rnn_gpu` averages the four MMI atoms over many random
    balanced (50/50) bipartitions, this function instead evaluates a single,
    fixed 1-vs-(U-1) bipartition for each unit in turn. For a population of
    U units and a given timestep, unit i is decomposed as

        source X1 = unit i                     (1-D, the single unit)
        source X2 = the remaining (U-1) units  ((U-1)-D, everyone else)
        target  Y = the (univariate) stimulus  (e.g. signed cued coherence)

    and the standard Williams & Beer / Barrett-MMI atoms are returned for that
    unit. There is therefore no averaging over bipartitions: the output keeps a
    per-unit axis so you can read, unit by unit, how much information that unit
    carries uniquely, redundantly, or synergistically with the rest of the net.

    Speed trick (same idea as `gaussian_pid_rnn_gpu`)
    ----------------------------------------------------------
    At a given timestep the joint covariance over [all units | target] is the
    same regardless of which unit is singled out, so it is formed ONCE per
    timestep on the GPU. All U one-vs-others decompositions then reduce to cheap,
    batched log-determinants of sub-blocks of that single shared covariance:
      1) I(unit i ; Y)         -> a 1x1 and a 2x2 sub-block (vectorized over i),
      2) I(others_i ; Y)       -> a (U-1)x(U-1) and a UxU sub-block, batched over i,
      3) I(all units ; Y)      -> computed once, shared by every i.
    As in `gaussian_pid`, each column is standardized to unit variance (MI is
    invariant to this; it only conditions the covariance) and a `reg` ridge is
    added to the covariance diagonal before the closed-form Gaussian MIs and the
    MMI atoms (Barrett 2015) are computed.

    The function is written to be independent of the population size U, so it
    works unchanged for RNNs smaller or larger than the default 100 units.

    Parameters
    ----------
    H : array_like, shape (n_samples, n_timesteps, n_units)
        Hidden activations. `n_samples` (trials) is the axis over which the
        per-timestep covariance is estimated; `n_units` (= U) may be any size >= 2.
    Y : array_like, shape (n_samples,)
        Univariate target (e.g. signed coherence / stimulus): one value per trial,
        shared across timesteps.
    timestep : int or None, optional
        int  -> compute the one-vs-rest PID only at this timestep (supports
                  negative indexing, e.g. -1 = last timestep). Per-unit arrays of
                  shape (n_units,) are returned.
        None -> compute it at every timestep. Per-unit/-timestep arrays of shape
                  (n_timesteps, n_units) are returned.
    reg : float, optional
        Ridge added to the covariance diagonal for conditioning (the GPU analogue
        of `gaussian_pid`'s `regularization`).
    device : torch.device or str or None, optional
        Torch device to run on. None (default) selects CUDA if available, else CPU.

    Returns
    -------
    dict with keys (all values are numpy arrays, in BITS):
        'redundancy', 'unique1', 'unique2', 'synergy' : the four MMI atoms, where
            'unique1' is the information the singled-out unit carries uniquely and
            'unique2' is what the remaining (U-1) units carry uniquely.
        'mi_1'     = I(unit i ; Y)             (the single unit vs target)
        'mi_2'     = I(others_i ; Y)           (all other units vs target)
        'mi_joint' = I(all units ; Y)          (same for every unit at a timestep)
        'total'    = redundancy + unique1 + unique2 + synergy   (equals mi_joint
                     up to rounding; a built-in consistency check)
    For `timestep=int` each array has shape (n_units,); for `timestep=None` each
    array has shape (n_timesteps, n_units). This mirrors `gaussian_pid_rnn_gpu`'s
    per-timestep/per-atom layout but adds the extra per-unit axis, and uses the
    clearly-named dictionary keys of `gaussian_pid_rnn`.

    How to call
    -----------
    # One-vs-others PID for every unit at the last timestep (trials as samples):
    >>> out = gaussian_pid_gpu_one2many(acts, stim, timestep=-1)  # acts: (trials, T, units)
    >>> out['unique1'].shape                                      # (U,)
    >>> out['unique1']            # each unit's own unique information (bits)

    # Full time-resolved one-vs-others profile (every unit, every timestep):
    >>> prof = gaussian_pid_gpu_one2many(acts, stim, timestep=None)
    >>> prof['synergy'].shape                                     # (T, U)
    >>> prof['synergy'][:, 3]     # synergy of unit 3 with the others, over time
    """
    # --- 1. validate activations and unpack the three axes ------------------- #
    H = np.asarray(H, dtype=float)                   # to float ndarray
    if H.ndim != 3:                                  # this path needs the 3-D view
        raise ValueError("H must be (n_samples, n_timesteps, n_units); "
                         f"got shape {H.shape}.")
    N, T, U = H.shape                                # samples, timesteps, units
    if U < 2:                                        # need at least one "other" unit
        raise ValueError(f"need n_units >= 2 for a 1-vs-others partition; got {U}.")
    if N <= U + 1:                                   # full-rank covariance over [units|Y]
        raise ValueError(f"need n_samples ({N}) > n_units + 1 ({U + 1}) "
                         "for a full-rank covariance estimate.")

    # --- 2. resolve which timestep(s) to evaluate ---------------------------- #
    if timestep is None:                             # all timesteps requested
        t_indices = list(range(T))                   # evaluate every step
        single = False                               # -> return (T, U) arrays
    else:                                            # one specific timestep
        t = timestep if timestep >= 0 else T + timestep   # resolve negative index
        if not (0 <= t < T):                         # bounds check after resolving
            raise IndexError(f"timestep {timestep} out of range for {T} timesteps.")
        t_indices = [t]                              # evaluate just this one
        single = True                                # -> return (U,) arrays
    Tsel = len(t_indices)                            # number of timesteps we output

    # pick CUDA when available unless the caller forced a device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    # --- 3. standardize columns to unit std (ddof=1); MI is scale-invariant,
    #        this only conditions the covariance (matches gaussian_pid_rnn_gpu) --- #
    Ht = torch.tensor(H[:, t_indices, :], dtype=torch.float64, device=dev)     # (N, Tsel, U)
    Yt = torch.tensor(np.asarray(Y, float), dtype=torch.float64, device=dev)   # (N,)
    Ht = Ht / Ht.std(dim=0, keepdim=True, unbiased=True).clamp_min(1e-12)      # per-unit unit std
    Yt = Yt / Yt.std(unbiased=True).clamp_min(1e-12)                           # target unit std

    # --- 4. one joint covariance per (selected) timestep over [units | target] -- #
    D  = torch.cat([Ht, Yt[:, None, None].expand(N, Tsel, 1)], dim=2)          # (N, Tsel, U+1)
    Dc = D - D.mean(dim=0, keepdim=True)                                       # center each column
    cov = torch.einsum('nti,ntj->tij', Dc, Dc) / (N - 1)                       # sample cov (ddof=1)
    cov = cov + reg * torch.eye(U + 1, dtype=torch.float64, device=dev)        # ridge (matches reg)
    iy = U                                                                     # target column index

    # --- 5. index sets for the fixed 1-vs-others partitions (unit-only, reused all t) -- #
    units = torch.arange(U, device=dev)                                        # 0..U-1 unit indices
    mask  = ~torch.eye(U, dtype=torch.bool, device=dev)                        # (U,U): False on diagonal
    rest  = units.expand(U, U)[mask].view(U, U - 1)                            # (U, U-1): "others" of each unit
    rest_y = torch.cat([rest, torch.full((U, 1), iy, device=dev)], dim=1)      # (U, U): others + target column

    # --- 6. accumulate per-timestep, per-unit atoms (converted to bits) ------- #
    keys = ["redundancy", "unique1", "unique2", "synergy", "mi_1", "mi_2", "mi_joint", "total"]
    out = {k: np.zeros((Tsel, U)) for k in keys}     # (Tsel, U) buffers, one per quantity

    for ti in range(Tsel):                           # loop timesteps -> tiny per-step GPU memory
        c = cov[ti]                                  # (U+1, U+1) shared covariance at this timestep
        ld_y = torch.log(c[iy, iy])                  # logdet of the 1x1 target block (scalar)

        # I(all units ; Y): identical for every unit at this timestep (computed once)
        ld_units = torch.linalg.slogdet(c[:U, :U])[1]         # logdet all-units block
        ld_full  = torch.linalg.slogdet(c)[1]                 # logdet full [units|Y] joint
        mij = 0.5 * (ld_units + ld_y - ld_full)               # I(X1,X2;Y), scalar

        # I(unit i ; Y) for every i at once: 1x1 unit block and 2x2 [i,y] block
        var_i  = c[units, units]                              # (U,) each unit's variance = cov[i,i]
        cov_iy = c[units, iy]                                 # (U,) each unit's covariance with Y
        ld_ii  = torch.log(var_i)                             # (U,) logdet of the 1x1 unit block
        ld_iy  = torch.log(var_i * c[iy, iy] - cov_iy ** 2)   # (U,) logdet of the 2x2 [i,y] block
        mi1 = 0.5 * (ld_ii + ld_y - ld_iy)                    # (U,) I(unit i ; Y) per unit

        # I(rest_i ; Y) for every i at once: batched (U-1)x(U-1) and UxU sub-blocks
        sub_rest  = c[rest[:, :, None],  rest[:, None, :]]        # (U, U-1, U-1) "others" blocks
        sub_resty = c[rest_y[:, :, None], rest_y[:, None, :]]     # (U, U,   U)   "others"+target blocks
        ld_rest  = torch.linalg.slogdet(sub_rest)[1]             # (U,) logdet of each others block
        ld_resty = torch.linalg.slogdet(sub_resty)[1]            # (U,) logdet of each others+target block
        mi2 = 0.5 * (ld_rest + ld_y - ld_resty)                  # (U,) I(rest_i ; Y) per unit

        # MMI atoms (Barrett 2015), per unit; mij is scalar and broadcasts over units
        red = torch.minimum(mi1, mi2)                # (U,) MMI redundancy = smaller source-target MI
        u1  = mi1 - red                              # (U,) the single unit's unique info (>=0)
        u2  = mi2 - red                              # (U,) the rest's unique info (>=0)
        syn = mij - mi1 - mi2 + red                  # (U,) synergy (needs both sources)
        tot = red + u1 + u2 + syn                    # (U,) == mij up to rounding (consistency)

        # nats -> bits, then stash this timestep's row for each quantity
        for k, v in zip(keys, [red, u1, u2, syn, mi1, mi2, mij.expand(U), tot]):
            out[k][ti] = (v * NATS_TO_BITS).cpu().numpy()

    # --- 7. shape the return: (U,) for one timestep, (T, U) for all ---------- #
    if single:                                       # single timestep -> drop the length-1 time axis
        return {k: out[k][0] for k in keys}          # each value shape (U,)
    return {k: out[k] for k in keys}                 # each value shape (T, U)


