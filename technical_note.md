# Technical Note: Information Decomposition in Task-Trained RNNs

## 1. Motivation and Scientific Questions

It is not well understood how information processing in neural circuits relates to the structure and organization of their internal representations. In 2022, Luppi et al. studied the information structure across different brain areas and found that sensorimotor cortices are strongly redundant, while association cortices are strongly synergistic. This suggests that different functional and integration demands drive different information structures, though the biological mechanisms remain unclear.

Previously, Mante et al. (2013) studied context-dependent decision-making in the prefrontal cortex (PFC)—an association area—by comparing macaque monkey PFC activity with recurrent neural networks (RNNs) trained on high-integration demand tasks. They found that context-dependent decisions require a recurrent, population-level solution that cannot be decomposed into independent units. While this is the conceptual definition of synergy, they did not quantify this using formal information-theoretic measures. 

Bringing these two perspectives together, we aim to answer the primary research question: **Does the high integration demand of context information lead to synergy in neural representations?**

### Background: Partial Information Decomposition (PID)
Classical information-theoretic measures, such as mutual information (MI) or Fisher information, quantify *how much* information a neural network carries in total, but not *how that information is distributed* across its units. 

Partial Information Decomposition (PID) addresses this gap. It takes the total mutual information that two sources, $X_1$ and $X_2$ (e.g., two different sets of neurons in the RNN), jointly carry about an input stimulus $Y$, denoted as $I(X_1, X_2 ; Y)$, and separates it into:
* **Redundant information:** Information processed in parallel by both sources.
* **Synergistic information:** Information that requires the joint activity of multiple neurons to decode.
* **Unique information:** Information about the stimulus encoded by only one of the sources.

We utilize the Minimal Mutual Information PID (MMI-PID), consistent with Luppi et al. (2022). MMI-PID defines redundancy as equal to the source that carries the least information about the stimulus:
$$Red = \min(I(X_1 ; Y), I(X_2 ; Y))$$
Once redundancy is fixed, the unique information follows as $I(X_i ; Y) - Red$, and synergy is the remaining joint information.

### Hypotheses and Predictions
We hypothesize that the PID profile of a task-trained RNN at the end of the stimulus period is dictated by the integration demand of the task.

* **Prediction 1:** Mutual Information (and subsequently synergy and redundancy) will peak at the end of the stimulus period. In RNNs, evidence accumulation culminates right before a decision is made, maximizing the total MI the network carries about the stimulus.
* **Prediction 2:** The Context task will lead to a more synergistic RNN, while the Perceptual task will lead to a more redundant RNN. Combining multiple input streams (context cue + stimulus) forces the network to encode information synergistically, as it cannot be decoded by any single unit reading all channels. Conversely, single-stream accumulation (Perceptual task) can be encoded redundantly in parallel.

## 2. Methodology

### 2.1 Experimental Design
We train artificial neural networks on the two tasks described by Mante et al. (2013) using the NeuroGym framework:
* **Perceptual Task (`ContextDecisionMaking-v0` masked):** Low integration demand. The network accumulates a single evidence stream.
* **Context Task (`ContextDecisionMaking-v0`):** High integration demand. The network must use a context cue to selectively integrate one of two conflicting sensory streams.
* **Training Objective:** All networks are trained using a standard supervised Cross-Entropy loss on the final decision period.

### 2.2 Network Architecture & Hyperparameters
To ensure biological plausibility and representational stability, we employ a Continuous-Time RNN (CTRNN):

$$h_{t+\Delta t} = h_t + \frac{\Delta t}{\tau}\Big[-h_t + W_{rec}\tanh(h_t) + W_{in}\,u_t + b\Big]$$

$$y_t = W_{out}\,h_t$$

* **Continuous-Time Dynamics:** We selected a CTRNN over a standard discrete-time Elman RNN. The time constant $\tau = 100$ ms acts as a biological low-pass filter, which is necessary for the network to successfully integrate the high-variance noisy stimulus ($\sigma = 1.0$) rather than tracking instantaneous noise fluctuations of the $\Delta t=10$ ms signal.
* **Network Parameters:** We use a hidden size of $N=100$. This specific hyperparameter optimizes the balance between successful, stable task convergence and representational analysis.  The network is fully connected without sign constraints (no Dale's law). Training uses BPTT with the Adam optimizer and gradient-norm clipping. 

### 2.3 Information-Theoretic Analysis
We compute PID on the hidden-state activations $h_t$ using the final discrete decision label as the target variable. To keep the decomposition tractable, units are coarse-grained into two functional subpopulations. Inference uses permutation tests (shuffling condition labels to build the null) and bootstrap confidence intervals, with Cohen's $d$ effect sizes.

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
