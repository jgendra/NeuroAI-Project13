# Information Decomposition in Task-Trained RNNs

**Does the training objective shape *how* information is organized inside a recurrent neural network?**

A controlled methods-and-mechanisms study of how different normative training pressures
(efficient coding vs. predictive coding) push task-trained RNNs along the
**redundancy ↔ synergy** axis, measured with Partial Information Decomposition (PID) and
Integrated Information Decomposition (ΦID).

> Course project for **NeuroAI and Machine Learning in Neuroscience (Sommersemester 2026, TUM)** — based on
> *Project 12: Quantifying Information Flow in Recurrent Neural Networks*.

![status](https://img.shields.io/badge/status-in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.12-blue)
![framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)

---

## Motivation

Classical information-theoretic measures (mutual information, Fisher information) quantify
*how much* information a network carries, but not *how that information is distributed*
across its units. PID instead separates information about a target into **redundant**,
**unique**, and **synergistic** components, and ΦID extends this to information flow over
time. Recent work has shown that this distinction is biologically meaningful — for example,
sensory cortex is redundancy-dominated while association cortex is synergy-dominated
(Luppi, Mediano et al. 2022).

This project asks whether the *training objective itself* leaves a measurable fingerprint
on this decomposition in artificial RNNs. **This is a study in artificial systems — not a
direct claim about biological neural data.**

## Research questions

| # | Question |
|---|----------|
| **Q1** | Does the training objective shift task-trained RNNs along the redundancy–synergy axis *at all*? (sanity check) |
| **Q2** | Does an efficient-coding activity penalty push the network toward **synergy** vs. vanilla supervised training? |
| **Q3** | Does a predictive loss have a **similar or distinct** effect on the redundancy–synergy balance? |
| **Q4** | Do these effects **depend on how much integration the task demands**? |

## Experimental design

A fully crossed design — **2 tasks × 3 conditions × 5 seeds = 30 RNNs**.

**Tasks** ([NeuroGym](https://neurogym.github.io/), `dt = 20 ms`):
- `PerceptualDecisionMaking-v0` — low integration demand (control)
- `ContextDecisionMaking-v0` — high integration demand (combine a context cue with the relevant stimulus stream)

**Training conditions** (cross-entropy masked to the decision period in all three):
1. **Vanilla supervised** — baseline: `L = CE`
2. **Activity-regularized** (efficient-coding analog) — `L = CE + λ·mean(h²)`, λ ≈ 1e-4
3. **Predictive auxiliary** (predictive-coding analog) — `L = CE + μ·MSE(W_pred·h(t), u(t+1))`, μ ≈ 0.1

> **Accuracy is matched across conditions before any comparison.** λ and μ are tuned so all
> conditions reach comparable task accuracy — otherwise a difference in information structure
> could just reflect a difference in task performance.

## Model

A continuous-time RNN (CTRNN), one recurrent layer, **80 units**, integrated with Euler steps:

$$h_{t+\Delta t} = h_t + \frac{\Delta t}{\tau}\Big[-h_t + W_{rec}\tanh(h_t) + W_{in}\,u_t + b\Big], \qquad y_t = W_{out}\,h_t$$

- τ = 100 ms, Δt = 20 ms (Δt/τ = 0.2)
- tanh recurrent activation; linear input and (logit) output layers
- Fully connected, no sign constraints (no Dale's law)
- Init: `W_rec` orthogonal (gain 1.0); `W_in`, `W_out`, `W_pred` Xavier uniform; biases 0
- Training: BPTT, Adam (lr 1e-3), batch 16, gradient-norm clipping at 1.0, ≤ 20k trials with early stopping

## Analysis

Computed on hidden-state activations `h(t)`:

- **PID** (stimulus / decision as target) — the core analysis for Q1–Q4. Units are coarse-grained into 2 functional subpopulations to keep the decomposition tractable.
- **ΦID** with `h(t+τ)` as target — *(stretch)* the dynamical question: does the network integrate information over time synergistically?
- **Pairwise MI** between units (KSG estimator) — correlation-structure sanity check.
- **Linear Fisher information** (bias-corrected) — *(optional)* likely redundant with the above.

**Statistics.** Because PID/ΦID estimates are bounded, skewed and non-Gaussian (and there are
only 5 seeds per condition), inference uses **permutation tests** (shuffling condition labels
to build the null) and **bootstrap** confidence intervals, with **Cohen's d** effect sizes
and **Bonferroni** correction for planned comparisons.

---

## Repository structure

```
NeuroAI-Project13/
├── README.md
├── environment.yml          # pinned conda environment
├── src/
│   ├── models/              # CTRNN definition + Euler integration
│   ├── training/            # training loop, loss functions, BPTT
│   └── analysis/            # PID / ΦID / MI / Fisher pipelines
├── notebooks/               # exploratory analysis + throwaway/example files
├── results/                 # saved model weights/activations, metric outputs
├── figures/                 # final figures
└── docs/                    # technical note, references
```

## Installation

```bash
# 1. Create the environment
conda env create -f environment.yml
conda activate neuroai-project13

# 2. Core dependencies (if not already pinned in environment.yml)
pip install torch neurogym numpy scipy matplotlib

# 3. Information-decomposition libraries
pip install git+https://github.com/Imperial-MIND-lab/integrated-info-decomp.git   # phyid
```

**JIDT (for PID/ΦID estimators) needs Java.** Install a JDK (Java 8+), download
`infodynamics.jar` from the [JIDT repo](https://github.com/jlizier/jidt), and bridge it from
Python with JPype:

```bash
pip install jpype1
# then point your code at the path of infodynamics.jar
```

## Usage

```bash
# Train the full sweep (2 tasks × 3 conditions × 5 seeds = 30 RNNs)
python -m src.training.run_sweep --config configs/sweep.yaml

# Run the information-theoretic analysis on trained networks
python -m src.analysis.run_pid --results results/

# Statistics + figures
python -m src.stats.run_tests --metrics results/metrics/
```

> **Compute note.** Everything runs on a laptop. The 80-unit CTRNNs train in minutes each
> (the sweep batches comfortably overnight). The real cost is the PID/ΦID estimation, which
> is why analysis is kept to 2 subpopulations and ΦID/Fisher are treated as stretch goals.

## Project roadmap

Planning and progress are tracked on the repo's **Projects** tab (Roadmap / Board / Table views).
Key dates: final roadmap **8 June**, midway presentation **17 June**, final presentation **15 July 2026**.

## References

- Williams & Beer (2010), *Nonnegative Decomposition of Multivariate Information* — [arXiv:1004.2515](https://doi.org/10.48550/arXiv.1004.2515)
- Mediano, Rosas et al. (2021), *Integrated Information Decomposition (ΦID)* — [doi:10.1073/pnas.2423297122](https://doi.org/10.1073/pnas.2423297122)
- Luppi, Mediano et al. (2022), *A synergistic core for human brain evolution and cognition* — [doi:10.1038/s41593-022-01070-0](https://doi.org/10.1038/s41593-022-01070-0)
- Mante, Sussillo, Shenoy & Newsome (2013), *Context-dependent computation by recurrent dynamics in PFC* — [doi:10.1038/nature12742](https://doi.org/10.1038/nature12742)
- Yang et al. (2019), *Task representations in neural networks trained to perform many cognitive tasks* — [doi:10.1038/s41593-018-0310-2](https://doi.org/10.1038/s41593-018-0310-2)
- Sussillo & Barak (2013), *Opening the black box: reverse-engineering RNN dynamics* — [doi:10.1162/neco_a_00409](https://doi.org/10.1162/neco_a_00409)
- Kanitscheider et al. (2015), *Measuring Fisher information accurately in correlated neural populations* — [doi:10.1371/journal.pcbi.1004218](https://doi.org/10.1371/journal.pcbi.1004218)
- Kraskov, Stögbauer & Grassberger (2004), *Estimating mutual information (KSG)* — [doi:10.1103/PhysRevE.69.066138](https://doi.org/10.1103/PhysRevE.69.066138)
- Lizier (2014), *JIDT: an information-theoretic toolkit* — [doi:10.3389/frobt.2014.00011](https://doi.org/10.3389/frobt.2014.00011)

**Tools:** [NeuroGym](https://neurogym.github.io/) · [phyid](https://github.com/Imperial-MIND-lab/integrated-info-decomp) · [JIDT](https://github.com/jlizier/jidt)

## Acknowledgements

Developed for the NeuroAI & ML in Neuroscience course (Computational Neuroscience, TUM) taught by
Prof. Dr. Julijana Gjorgjieva and the teaching team.
