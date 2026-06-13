"""
gaussian_pid.py
===============

Gaussian analytic Partial Information Decomposition (PID), using the
Minimum-Mutual-Information (MMI) redundancy of Barrett (2015), i.e. the same
Gaussian machinery that the `phyid` library (Mediano/Rosas/Luppi) uses for its
Integrated Information Decomposition (PhiID), but specialised to *static* PID:
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
* MMI forces one of the two unique terms to be exactly zero and forces
  redundancy to be the smaller of the two source-target MIs, regardless of how
  the sources actually relate. This is a rigidity of the MMI definition.
* Barrett's exactness result is for a *univariate* target. The code accepts a
  multivariate target mechanically, but the MMI interpretation is cleanest for a
  univariate target (which is what this project uses: signed coherence /
  stimulus value).
* The Gaussian assumption is only an approximation for bounded tanh activations.
"""

import numpy as np


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


# ----------------------------------------------------------------------------- #
# Core analytic PID (the function the whole project is built on)
# ----------------------------------------------------------------------------- #
def gaussian_pid(sources_1, sources_2, target,
                 log_base="e", standardize=True, regularization=0.0):
    """
    Gaussian analytic MMI-PID for two source groups and one target.

    This is the atomic primitive. It treats every *row* as one observation
    (e.g. one trial) and every *column* as one variable (e.g. one RNN unit, or
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
          * (n_samples, n_timesteps, n_units)  -> per-timestep PID is available, or
          * (n_samples, n_units)               -> already a single snapshot.
        `n_samples` is the ensemble dimension over which covariance is estimated
        (in the real project: trials; for a single-trajectory test RNN: time, if
        you pass the trajectory in the samples axis).
    target : array_like
        The target / stimulus. Either
          * (n_samples, n_timesteps)  (a value per trial and timestep),
          * (n_samples,)              (one value per trial, shared across time), or
          * (n_samples, n_timesteps, dt) / (n_samples, dt) for a multivariate target.
    timestep : int or None, optional
        * int  -> compute PID only at this timestep (supports negative indexing,
                  e.g. -1 = last timestep). Returns scalar atoms.
        * None -> compute PID at every timestep. Returns arrays of shape
                  (n_timesteps,). Requires 3-D `activations`.
    bipartitions : {"random", "half"} or list of (idx1, idx2), optional
        How to split the units into two source groups:
          * "random" (default): draw `n_bipartitions` random splits.
          * "half": a single split into first-half / second-half units.
          * explicit list: each element is a pair (idx1, idx2) of index arrays.
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
