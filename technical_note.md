# Technical Note: Information Decomposition in Task-Trained RNNs

## 1. Motivation and Scientific Questions
Classical information-theoretic measures quantify *how much* information a network carries, but not *how that information is distributed* across its units. Partial Information Decomposition (PID) separates information about a target into **redundant**, **unique**, and **synergistic** components. Recent work demonstrates that this distinction is biologically meaningful (e.g., sensory cortex is redundancy-dominated while association cortex is synergy-dominated). 

This project investigates whether the *training objective itself* leaves a measurable fingerprint on this decomposition in artificial RNNs. We address the following research questions:
* **Q1:** Does the training objective shift task-trained RNNs along the redundancy–synergy axis?
* **Q2:** Does an efficient-coding activity penalty push the network toward synergy compared to vanilla supervised training?
* **Q3:** Does a predictive loss have a distinct effect on the redundancy–synergy balance?
* **Q4:** Do these effects depend on the cognitive integration demand of the task?

## 1. Motivation and Scientific Questions
Classical information-theoretic measures quantify *how much* information a network carries, but not *how that information is distributed* across its units. Partial Information Decomposition (PID) separates information about a target into **redundant**, **unique**, and **synergistic** components. 

This project investigates whether the *cognitive integration demand* of a task leaves a measurable fingerprint on this decomposition in artificial RNNs. Closely aligning with the seminal experiments of Mante et al. (2013), we address the following research questions:
* **Q1:** Does cognitive integration demand (low vs. high) shift task-trained RNNs along the redundancy–synergy axis?
* **Q2:** Why are continuous-time dynamics ($\tau$) biologically necessary for noise integration compared to discrete-time networks (e.g., Elman RNNs)?
* **Q3:** How does network capacity (specifically a hidden size of 100) optimize the balance between successful task convergence and representational analysis?

## 2. Methodology

### 2.1 Experimental Design
We utilize a fully crossed design comprising **2 tasks × 10 seeds = 20 RNNs**.
* **Tasks (NeuroGym, dt = 10 ms):**
  * `PerceptualDecisionMaking-v0`: Low integration demand (control).
  * `ContextDecisionMaking-v0`: High integration demand (requires dynamic routing of a context cue).
* **Training Objective:** All networks are trained using a standard supervised Cross-Entropy loss on the final decision period.


### 2.2 Model Architecture
We implement a continuous-time RNN (CTRNN) consisting of 80 hidden units integrated with Euler steps:

$$h_{t+\Delta t} = h_t + \frac{\Delta t}{\tau}\Big[-h_t + W_{rec}\tanh(h_t) + W_{in}\,u_t + b\Big]$$

$$y_t = W_{out}\,h_t$$

We use $\tau = 100$ ms and $\Delta t = 10$ ms. The network is fully connected without sign constraints (no Dale's law). Training uses BPTT with the Adam optimizer and gradient-norm clipping. 

### 2.3 Information-Theoretic Analysis
We compute PID on the hidden-state activations $h_t$ using the final discrete decision label as the target variable. To keep the decomposition tractable, units are coarse-grained into two functional subpopulations. Inference uses permutation tests (shuffling condition labels to build the null) and bootstrap confidence intervals, with Cohen's $d$ effect sizes.

---

## 3. Results

### 3.1 Network Architecture: Elman vs. CTRNN Dynamics
*(Summarize the findings from your tests where the Elman network failed to integrate noise while the CTRNN succeeded. Mention the biological low-pass filter effect of $\tau$.)*
* **Key Finding:** ...
* **Figure Reference:** `figures/learning_curves/elman_vs_ctrnn.png`

### 3.2 Impact of Hidden Size on Capacity and Information
*(Discuss how changing the hidden size affected the task performance and whether it forced the network into different representational regimes.)*
* **Key Finding:** ...
* **Figure Reference:** e.g. `figures/learning_curves/hidden_size_comparison.png`

### 3.3 PID Profiles: Cognitive Demand and Training Objectives
*(Describe the PID plots showing Synergy dominating in Context tasks and Redundancy dominating in Perceptual tasks. Then, discuss how the efficient/predictive coding objectives altered these baseline curves.)*
* **Key Finding (Task Demand):** ...
* **Key Finding (Training Objectives):** ...
* **Figure References:** `figures\mean_all_time_pid\mean_all_time_pid.png`, `figures\norm_pid_decision_time\decision_time_pid_norm_violin.png`

---

## 4. Limitations
* **Gaussian Approximation:** The analytic PID relies on multivariate Gaussian assumptions (covariance matrices), which is only an approximation for the bounded `tanh` activations of the RNN.
* **Coarse-Graining:** Due to the exponential scaling of the PID lattice, we were forced to split the 100 units into two macroscopic subpopulations. We miss micro-level synergistic interactions between individual neurons.
* **Attractor Dynamics vs. Continuous Targets:** Because the network forms discrete point attractors to solve the classification task, our PID estimator is blind to continuous stimulus magnitude prior to decision commitment.

## 4. Limitations
* **Gaussian Approximation & MI Cross-Check:** The analytic PID relies on multivariate Gaussian assumptions (covariance matrices), which is only an approximation for the bounded `tanh` activations of the RNN. *To validate this, we cross-checked the total Mutual Information using a non-parametric estimator (e.g., KSG or binning). Results showed [insert brief result: e.g., the Gaussian assumption underestimated absolute bits but preserved the relative geometry].*
* **1-vs-Many Neuron Split:** Due to the exponential scaling of the PID lattice, our primary analysis split the 100 units into two macroscopic subpopulations of 50. *We further extended this to a 1-vs-99 split to investigate micro-level synergistic interactions of individual neurons against the ensemble, revealing [insert brief result].*
* **Attractor Dynamics:** Because the network forms discrete point attractors to solve the classification task, our PID estimator is blind to continuous stimulus magnitude prior to commitment.

## 5. References
* Luppi, A. I., Mediano, P. A., Rosas, F. E., Holland, N., Fryer, T. D., O’Brien, J. T., ... & Stamatakis, E. A. (2022). A synergistic core for human brain evolution and cognition. Nature neuroscience, 25(6), 771-782. DOI: https://doi.org/10.1038/s41593-022-01070-0.
* Mante, V., Sussillo, D., Shenoy, K. V., & Newsome, W. T. (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. Nature, 503(7474), 78-84. DOI: https://doi.org/10.1038/nature12742.
### 5.1. Methods references
* Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information. arXiv preprint arXiv:1004.2515. DOI: https://doi.org/10.48550/arXiv.1004.2515.
* Barrett, A. B. (2015). Exploration of synergistic and redundant information sharing in static and dynamical Gaussian systems. Physical Review E, 91(5), 052802. DOI: https://doi.org/10.1103/PhysRevE.91.052802.
