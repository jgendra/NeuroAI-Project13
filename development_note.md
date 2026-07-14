# Development & Course Notes

## Project Roadmap
Planning and progress are tracked on the repo's Projects tab.
* **8 June 2026:** Final roadmap formulation.
* **17 June 2026:** Midway presentation.
* **15 July 2026:** Final presentation and repository submission.

## Future / Stretch Goals
* **Integrated Information Decomposition (ΦID):** Extend the analysis over time using $h(t+\tau)$ as the target to determine if the network integrates information over time synergistically.
* **Pairwise MI:** Implement KSG estimators for correlation-structure sanity checks.
* **Linear Fisher Information:** Implement bias-corrected estimators to compare with classical methodologies.
* **Training Conditions (Matched for accuracy):** Analyze the influence on PID when changing the training loss
  1. *Vanilla supervised:* `L = CE`
  2. *Activity-regularized:* `L = CE + λ·mean(h²)`
  3. *Predictive auxiliary:* `L = CE + μ·MSE(W_pred·h(t), u(t+1))`

## Acknowledgements
This repository was developed for the **NeuroAI & Machine Learning in Neuroscience** course (Computational Neuroscience, TUM, Sommersemester 2026) taught by Prof. Dr. Julijana Gjorgjieva and the teaching team. Based on *Project 12: Quantifying Information Flow in Recurrent Neural Networks*.
