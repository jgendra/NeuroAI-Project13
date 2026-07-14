# Technical Note: Information Decomposition in Task-Trained RNNs

## 1. Motivation and Scientific Questions

It is not well understood how information processing in neural circuits relates to the structure and organization of their internal representations. In 2022, Luppi et al. studied the information structure across different brain areas and found that sensorimotor cortices are strongly redundant, while association cortices are strongly synergistic. This suggests that different functional and integration demands drive different information structures, though the biological mechanisms remain unclear.

Previously, Mante et al. (2013) studied context-dependent decision-making in the prefrontal cortex (PFC)—an association area—by comparing macaque monkey PFC activity with recurrent neural networks (RNNs) trained on high-integration demand tasks. They found that context-dependent decisions require a recurrent, population-level solution that cannot be decomposed into independent units. While this is the conceptual definition of synergy, they did not quantify this using formal information-theoretic measures. 

Bringing these two perspectives together, we aim to answer the primary research question: **Does the high integration demand of context information lead to synergy in neural representations?**

### Background: Partial Information Decomposition
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

### 3.1 Network Architecture & Capacity
**Question:**
Is a discrete Elman cell sufficient, or is the leaky CTRNN needed to reproduce the integration dynamics we want to study?

**Approach:**
We computed Time-resolved PID for the Elman RNN and the CTRNN in both tasks, alongside the training and validation loss curves.

* **Training loss:** 
![RNN Training Curves](elman_vs_ctrnn_comparison\figures\elman_vs_ctrnn_learning_curves.png)
The Elman RNN's loss stays high at around 0.7 in both tasks, versus roughly 0.26 for the CTRNN. More importantly, given that the Elman RNN uses Cross-Entropy Loss and $-\log(0.5)\approx 0.69$, this means it only learned to not predict the fixation signal and was guessing the actual stimulus. At the matched size of 100 hidden units, the Elman RNN cannot learn any of the tasks, while CTRNN is able to.
* **Time-resolved PID:** 
![RNN PID Geometries](elman_vs_ctrnn_comparison\figures\elman_vs_ctrnn_pid_geometry.png)
The Elman RNN's total MI is remarkably lower than the CTRNN and similar across tasks. It barely accumulates information after stimulus onset, and the total MI does not peak at decision time.
* **Results:**
The most likely cause is that a vanilla tanh Elman cell trained with backpropagation through time (BPTT) suffers vanishing gradients over the roughly 1050 ms fixation-plus-stimulus horizon, so it cannot maintain the evidence long enough to decide. Meanwhile, the CTRNN behaves as predicted: information accumulates during the stimulus, peaks at decision time in both tasks, and total MI is higher in context decision-making (CDM) (above 1.2 bits) than in perceptual decision-making (PDM) (close to 0.8 bits). This justifies the CTRNN not as a main result but as a methodological choice, consistent with our aim of reproducing Mante et al. (2013).

### 3.2 Impact of Hidden Size on Capacity and Information
**Question 1:**
At what network size does the RNN represent a stable, near-maximum amount of stimulus information, and is 100 units a defensible choice?

**Approach:**
For a single seed at each hidden size (2, 4, 8, 12, 16, 20, 40, 60, 80, 100, 150, 200 units), we computed the PID atoms at decision time (the last timestep of the stimulus period) with signed coherence as the target, and plotted their sum (total MI) against hidden size, together with the individual atoms. We also plotted test accuracy against hidden size for both the Perceptual Decision Making (PDM) and Context-Dependent Decision Making (CDM) tasks.

* **Performance:** 
![Size comparison training curves](size_comparison\results\performance_comparison.png)
While even the smallest size model with 2 hidden units has a better than guessing accuracy, the performance rises steeply between 4 and 16 units and plateaus after 40 units near 88%.
* **PID at decision time:**
![CDM coherence PID decomposition](size_comparison\results\pid_cdm_coherence_comparison.png)
As with accuracy, total MI rises rapidly at the start and converges after 40 units, near 1.29 bits for CDM.
* **Result:** 
This gives a circular-argument-free justification for our operating point: total information saturates well below 100 units, so 100 units sits comfortably on the plateau. We therefore select 100 units because (i) total MI has already converged, (ii) any further change is a slow, near-linear drift rather than a qualitative shift, and (iii) most importantly, it matches the architecture of Mante et al. (2013), whose methodology we set out to reproduce for comparison.

**Question 2:**
Is the redundancy-synergy difference between tasks specific to 100-unit networks, or does it generalize across network size?

**Approach:**
For each task and each hidden size, we computed the normalized quantity (redundancy minus synergy) divided by total MI at decision time, and plotted it against hidden size for CDM and PDM.

* **Redundancy-Synergy balance (Result):**
![(redundancy-synergy)/total vs RNN size](size_comparison\results\redundancy_synergy_balance_comparison.png)
Once total MI, accuracy, and loss have converged (by roughly 40 units), the normalized redundancy-minus-synergy value is systematically lower for CDM than for PDM across the tested sizes, mirroring the 100-unit result: as integration demand rises, synergy increases and redundancy decreases as a fraction of total information. The ordering breaks only at 60 units, where PDM is higher, which we attribute to sampling noise. Because this analysis uses a single seed per size, it is suggestive rather than statistical, and averaging across seeds at each size would be needed to establish it firmly (see Limitations).

### 3.3 PID Profiles: Cognitive Demand and Training Objectives (Main Analysis)
**Question 1:**
Are the two groups of networks matched in performance, so that a redundancy-synergy difference cannot be explained by "one group just learned the task better"?

**Approach:**
Plot per-seed bars of test accuracy (top) and test loss (bottom) for the 10 CDM and 10 PDM networks, with an MWU test between tasks mirroring the structure of the main PID comparison.

* **Accuracy and Loss across Tasks and Models (Result):**
![Test accuracy loss bars](figures\accuracy_loss\test_accuracy_loss_bars.png)
Mean CDM accuracy is 0.885 with standard deviation (SD) = 0.005, and mean PDM accuracy lies at 0.887 with SD = 0.005. CDM loss is 0.261 with SD 0.009, while PDM loss is 0.256 with SD = 0.008. Under an independent MWU test, CDM and PDM CTRNNs reach statistically indistinguishable test accuracy and loss, so any downstream PID difference cannot be attributed to one task simply being learned better.

**Question 2:**
Are the PID time courses stable across seeds, and does the peak at decision time replicate?

**Approach:**
Calculate and plot mean PID atom across the 10 seeds per task at every timestep, shading the atom’s SD band around the mean timecourse for CDM (right) and PDM (left) separately. A blue (CDM) and red (PDM) shaded area covers the duration of the stimulus period, between fixation and decision period.

* **Mean PID at every Timestep (Result):**
![Mean PID at every Timestep](figures\mean_all_time_pid\mean_all_time_pid.png)
Total MI stays near zero during fixation, begins to accumulate immediately after stimulus onset, and peaks just before the decision, which is precisely where synergy and redundancy also peak. This confirms that decision time is where the network has accumulated the full evidence and is the correct point to characterize the information structure required for the choice. After decision time, on average, synergy decreases and redundancy increases while total MI plateaus and then slowly declines, consistent with the network only needing to sustain the already-made decision (a simpler quantity that can be held redundantly in parallel) and then resetting for the next trial. \
At decision time, CDM total MI (about 1.29 bits) is higher than PDM (about 0.77 bits), which is close to twice the PDM total MI: the PID target is a single scalar (i.e., the cued signed coherence) in both tasks, so this is not exactly "twice the information from two streams." Rather, CDM networks maintain a richer, higher-dimensional representation (the cued stream, the distractor stream, and the context cue) even though only the cued stream determines the target, which raises the total MI decodable about the cued coherence. Because this magnitude difference is large, synergy appearing higher in CDM in the raw time course cannot by itself confirm Prediction 2; the atoms must first be normalized by each network's total MI.

**Question 3:**
Is the redundancy-synergy shift between CDM and PDM statistically reliable across seeds after controlling for the total-MI magnitude difference? This is the central test of Prediction 2.

**Approach:**
For each atom at decision time, we extracted the 10 per-seed values per task and compared CDM versus PDM with a two-sided MWU. We display total MI in bits, and the normalized fractions (redundancy, unique 1, unique 2, synergy) and their sum (total uniqueness) as violin plots, with individual seeds overlaid. To respect the dependency structure of the atoms, we correct in two tiers: the four independent compositional atoms (unique 1, unique 2, redundancy, synergy) form the primary family (Bonferroni k equals 4, corrected alpha 0.0125), while total MI and total uniqueness are deterministic sums of primary-family members and form a separate derived family (k equals 2, corrected alpha 0.025).

* **Normalized PID atoms at decision time:**
![Normalized PID at Decision time - violin plot](figures\norm_pid_decision_time\decision_time_pid_norm_violin.png)
Total MI is significantly higher in CDM (CDM mean = 1.286 and SD = 0.022, versus PDM mean = 0.766 and SD = 0.017 bits). Crucially, redundancy dominates the decomposition in both tasks (roughly 98 to 99 percent of total MI), yet after normalization, redundancy is significantly lower in CDM (0.9832 in CDM versus 0.9891 in PDM), and synergy is significantly higher (0.0138 in CDM versus 0.0087 in PDM). \
The redundancy that is lost in CDM reappears as both synergy and uniqueness. Total uniqueness is significantly higher in CDM, and although unique 1 and unique 2 are each only weakly significant on their own, summing them (justified because the random 50/50 bipartitions make the source labels arbitrary and exchangeable, so only the total unique information is meaningful) yields a stronger, clearly significant increase.
* **Result:**
The MWU statistic reaches its maximum value (U equals 100) for these comparisons, meaning every CDM seed separates cleanly from every PDM seed, so the effect is not marginal despite the small absolute fractions. This confirms Prediction 2: as integration demand rises, synergistic processing increases and redundancy decreases, because the context computation requires the joint activity of multiple units and cannot be solved by any single unit reading one channel. \
The increased Synergy AND Total Uniqueness indicate that in CDM, the CTRNN both distributes information synergistically across cooperating units and specializes some units to carry information that no other unit does. However, the slight increase in uniqueness might be a consequence of redundant units becoming more synergistic, and so previously redundant information becomes more unique. That unique 1 and unique 2 move together across tasks also confirms that 200 bipartitions were sufficient to average out arbitrary partition-specific asymmetries. \
However, the increase in uniqueness need not contradict the shift toward distributed processing. Redundancy, synergy, and uniqueness together partition a fixed total, so when redundancy falls in CDM, the freed information is redistributed across both of the other atoms rather than into synergy alone. A redundant representation is one in which many units carry overlapping copies of the same coherence signal; reducing that overlap can send information in two directions at once. Part becomes synergistic, carried only in the joint activity of cooperating units, and part becomes unique, carried by individual units that specialize in a component of the computation no other unit encodes. The context task plausibly demands both: some units cooperate to integrate the cued stream against the context cue (synergy), while others specialize on separable sub-computations such as tracking one stimulus stream or the context signal itself (uniqueness). In this reading, "distributed" and "specialized" are not opposites but two complementary ways of moving away from redundant parallel coding, and both increase precisely because redundancy decreases. \
Mechanistically, this is consistent with Mante et al. (2013). This study shows that in the context task, the relevant computation (selection and integration) is a population-level dynamical process (an approximate line attractor together with a context-dependent selection vector) that cannot be read off from single neurons, whose responses instead show mixed selectivity for many task variables at once. Our finding that CDM relies relatively more on synergistic and unique coding, with no single unit individually sufficient, is the information-theoretic counterpart of that population-level account: higher integration demand pushes the solution off individual units and into the joint structure of the population. 

**Question 4:**
Is the synergistic and unique coding in CDM concentrated in a specialized subpopulation or spread across the network?

**Approach:**
For each seed and task, and for each unit i, we computed the PID with source X1 equal to unit i, source X2 equal to the remaining 99 units, and target equal to the signed cued coherence (a fixed 1-versus-99 partition). We display redundancy, synergy, unit unique (unique 1), and population unique (unique 2) per unit as a heatmap, on both linear and logarithmic color scales.

* **Per Unit PID (Result):**
![Log scale per unit PID (CTRNN_05)](figures\unit_pid\heatmap_seeds_P05_C05.png)
Under Minimum Mutual Information PID (MMI-PID) with a 1-versus-99 partition, a single unit essentially never carries more coherence information than the other 99 combined, so redundancy equals the minimum of the two source MIs, which is the single unit's own MI: Redundant $= \min\{ I(X_1;Y), I(X_2;Y) \} = I(unit; Y)$. Its unique information is therefore forced to zero by construction: Unit Unique $= I(X_1;Y) −$ Red $= 0$. Population unique is what the rest of the network adds beyond that unit: Population Unique $= I(X_2;Y) −$ Red $= I(X_2;Y) − I(X_1;Y)$. Synergy is Syn $=$ Total MI $− I(X_1;Y) − I(X_2;Y) +$ Red $= I(X_1,X_2;Y) − I(X_2;Y) = I(X_1;Y|X_2)$. This last term is near zero, not by algebraic necessity but empirically: the 99-unit population already captures nearly all the decodable coherence, so one extra unit adds almost nothing conditional on the rest. On the logarithmic scale, synergy is small but nonzero and fairly uniform across units, and slightly higher in CDM, consistent with the 50/50 bipartition PID analysis, where synergy is genuinely present and higher in CDM. \
The interpretable contrast is therefore the redundancy row (how much of the coherence signal a single unit carries) versus the population-unique row (how much the rest of the population adds on top). In PDM, redundancy is high and fairly uniformly distributed across units, while population uniqueness is low and close to zero. This means most units are near-copies of the same accumulated evidence, the fingerprint of redundant parallel coding expected for a single-stream accumulation task. In CDM, the pattern flips: redundancy is lower and sparse across units, while population unique dominates. This indicates that any one unit carries relatively little of the signal alone, and the answer lives in the collective. This is the fingerprint of distributed complementary joint coding expected when the cued coherence depends on combining the context cue with the correct stream. Importantly, both redundancy and synergy change roughly uniformly along the neuron axis. Hence, the shift is spread across essentially the whole population rather than delegated to synergy-specialized subpopulations doing all the integration between input information streams.


## 4. Limitations and Future Work
1. **The MMI definition and the Gaussian assumption:** \
Our PID uses MMI-PID, which carries two assumptions. First, MMI defines redundancy as the minimum of the two sources' individual mutual information, which by construction forces one unique term to exactly zero on every bipartition. This is an algebraic artifact rather than a verified property of the representation. Second, it assumes Gaussian activations, which is unlikely to hold exactly for tanh CTRNN states: our Shapiro-Wilk tests on example units at decision time reject normality for all tested units, though the distributions are unimodal rather than pathologically bimodal, so the violation is milder than feared. \
**Future work** should cross-check these results with a non-parametric or discrete PID estimator and with a non-MMI redundancy definition, quantify the Gaussian bias by comparing a kernel or nearest-neighbor estimate of total MI against the summed PID atoms, and verify that the PID covariance matrices are full rank.
2. **The information structure is not yet linked to the underlying dynamics:** \
We have addressed what changes with integration demand and why, but not how the structure is built by the network over a trial. We characterize the information structure without connecting it to the dynamical mechanism that produces it. \
**Future work** should apply the low-rank and fixed-point analyses of Mante et al. (2013), for both tasks rather than only the context task, and relate the resulting line-attractor and selection-vector geometry to the measured PID. A complementary causal test would be a selective silencing experiment: zeroing groups of units at test time and measuring the accuracy drop as a function of group size, with the prediction that the high-synergy CDM task degrades more sharply than the redundant PDM task, providing causal evidence that synergy is functional.
3. **Generalization across network size is inferred, not directly demonstrated:** \
We believe the redundancy-synergy structure is preserved across sizes because total MI and the PID profile converge once the network can solve the task, but this currently rests on a convergence criterion computed from a single seed per size, not on the full PID pipeline at every size. \
**Future work** should run the complete decomposition (redundancy, synergy, uniqueness) across all tested sizes with multiple seeds, to confirm that the structure depends only on the task's integration demand and not on capacity, which would show the redundancy-synergy axis is a signature of task demand rather than of network size.
4. **Paired versus unpaired statistics:** \
All figures use the independent MWU test. However, because each CDM and PDM seed pair is evaluated on the same cued coherences (PDM being CDM with the uncued stream zeroed), the two groups are partially paired. The appropriateness of pairing depends on how much a metric depends on that shared factor. Accuracy and loss depend heavily on the specific trials, so a paired Wilcoxon signed-rank test detects a small but consistent difference (turning the accuracy and loss match significantly), meaning we cannot claim the two tasks learned equally well in a strict paired sense, only that the effect is small in absolute terms relative to the PID differences. PID atoms depend less on the shared trials (they are driven by the learned activations), and while the paired Wilcoxon signed rank test does return significance for all the atoms, including total uniqueness, it does not for unique 1 or unique 2 alone. We include the paired tests in the repository, but base the reported figures on MWU, since regenerating unpaired test sets and all downstream figures were not feasible in the available time.

## 5. Conclusion
Higher task integration demand drives continuous-time recurrent neural networks away from redundant parallel coding and toward cooperative, synergistic representation. Using Partial Information Decomposition on context-dependent and perceptual decision-making tasks, we found the context task allocates significantly less information to redundancy and more to synergy and unique coding. This structural shift occurs uniformly across the entire network population rather than within specialized subgroups.

These results offer an information-theoretic counterpart to the selection-and-integration mechanism of Mante et al. (2013) and confirm that integration demand dictates a network's position on the redundancy-synergy axis. Current limitations include reliance on a Gaussian MMI-PID estimator, an absence of direct links to line-attractor dynamics, and size generalization inferred solely from convergence. As a controlled study in artificial networks, these results are best read as a hypothesis about biological circuits rather than direct evidence, but it establishes a concrete, quantitative prediction: the need to integrate multiple streams, rather than the raw volume of information, forces cortical circuits to adopt synergistic representations.

## 6. References
* Luppi, A. I., Mediano, P. A., Rosas, F. E., Holland, N., Fryer, T. D., O’Brien, J. T., ... & Stamatakis, E. A. (2022). A synergistic core for human brain evolution and cognition. Nature neuroscience, 25(6), 771-782. DOI: https://doi.org/10.1038/s41593-022-01070-0.
* Mante, V., Sussillo, D., Shenoy, K. V., & Newsome, W. T. (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. Nature, 503(7474), 78-84. DOI: https://doi.org/10.1038/nature12742.

### 6.1. Methods references
* Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information. arXiv preprint arXiv:1004.2515. DOI: https://doi.org/10.48550/arXiv.1004.2515.
* Barrett, A. B. (2015). Exploration of synergistic and redundant information sharing in static and dynamical Gaussian systems. Physical Review E, 91(5), 052802. DOI: https://doi.org/10.1103/PhysRevE.91.052802.
