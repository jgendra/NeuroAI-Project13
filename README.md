# Information Decomposition in Task-Trained RNNs

![status](https://img.shields.io/badge/status-in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.11-blue)
![framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)

## 1. Project summary
This project tests whether different normative training objectives shape how information is distributed within recurrent neural networks. We hypothesize that efficient-coding and predictive-coding pressures push task-trained continuous-time RNNs (CTRNNs) along the redundancy-synergy axis depending on the task's integration demands. To test this, we compare vanilla supervised, activity-regularized, and predictive-auxiliary CTRNNs trained on two NeuroGym tasks. We evaluate the networks' internal representations using Partial Information Decomposition (PID) to quantify redundant, unique, and synergistic information.

## 1. Project summary
This project tests whether cognitive integration demands shape how information is distributed within recurrent neural networks. Closely replicating the experimental design of Mante et al. (2013), we train continuous-time RNNs (CTRNNs) on two NeuroGym tasks with differing integration requirements (Perceptual vs. Context). We evaluate the networks' internal representations using Partial Information Decomposition (PID) to quantify redundant, unique, and synergistic information. Our analysis compares these information profiles to understand how normative task pressures alter representational geometry.

## 1. Project summary
This project tests whether cognitive integration demands shape how information is distributed within recurrent neural networks. Closely replicating the experimental design of Mante et al. (2013), we train continuous-time RNNs (CTRNNs) on two NeuroGym tasks with differing integration requirements. We evaluate the networks' internal representations using Partial Information Decomposition (PID) to quantify redundant, unique, and synergistic information. Additionally, we cross-validate our Gaussian PID estimator against non-parametric Mutual Information methods to assess the validity of the Gaussian approximation in bounded `tanh` networks.

## 1. Project summary
This project tests whether cognitive integration demands shape how information is distributed within recurrent neural networks. Closely replicating Mante et al. (2013), we train continuous-time RNNs (CTRNNs) on tasks with differing integration requirements and evaluate them using Partial Information Decomposition (PID). We cross-validate our Gaussian PID estimator against non-parametric Mutual Information methods to assess the validity of the Gaussian approximation. Finally, we expand our decomposition from macroscopic subpopulations to a 1-vs-many neuron split to investigate micro-level synergistic interactions.

## 2. Repository structure
```text
NeuroAI-Project13/
├── README.md                # Project overview and run instructions
├── technical_note.md        # Detailed methodology, results, and limitations
├── environment.yml          # Pinned conda environment
├── requirements.txt         # Pip dependencies
├── pyproject.toml           
├── src/                     # Source code (models, training, tasks, analysis)
├── notebooks/               # Exploratory scratch analysis
├── results/                 # Saved model weights, activations, and PID outputs
├── figures/                 # Generated plots for the presentation and technical note
└── docs/                    # Additional references and literature
```

## 3. How to run

### Environment Setup
This project requires **Python 3.11+**. The source code is structured as a module, meaning you must install the project in editable mode (`-e .`) to allow absolute imports (e.g., `import src.models`) inside notebooks and scripts.

Choose the installation method that matches your hardware:

#### Option A: Conda (Recommended for GPU users)
Conda automatically resolves the correct PyTorch CUDA binaries.
```bash
conda env create -f environment.yml
conda activate neuroai-project13
pip install -e .
```

#### Option B: Pip (NVIDIA GPU / CUDA 12.1)
If you are using a Linux/Windows machine with an NVIDIA GPU and prefer pip:
```bash
# Installs PyTorch via the explicit CUDA 12.1 index
pip install -r requirements.txt
pip install -e .
```

#### Option C: Pip (CPU Only - Mac / No GPU)
If you are running this on a standard laptop without an NVIDIA GPU:
```bash
# Installs PyTorch cpu-only version
pip install -e .
```

*Note: The Partial Information Decomposition (PID) analysis uses a custom-built, pure-Python Gaussian PID estimator included in the source code. No external Java dependencies or JIDT installations are required.*

### Execution & Figure Generation

0. **Dataset generation:** The datasets have to be generated to use the notebooks. They are pre-configured in `src/tasks/mante_config.py` and will be saved into the relative `data/[task]` directory. To reproduce our results run `python src/tasks/data_generator.py --mode context` and `python src/tasks/data_generator.py --mode perceptual`. If you use the provided model weights (without retraining) one can ommit the train and validation set generation by providing the extra parameters `--n_train 0 --n_val 0`. For further details have a look at `src/tasks/README_data.md`.
1. **Train the models:** Run the full sweep (20 RNNs) via notebook `notebooks/05_Full_train_pipeline.ipynb`. The weights will be saved to `results/model_weights/[task]` and are already delivered as pre-computed *(Expected runtime: ~2 hours on a standard GPU).*
2. **Compute PID:** Extract information metrics via notebook `notebooks/06_Full_PID_analysis_pipeline.ipynb`. *(Expected runtime: ~10 minute on CPU, <1 minute on GPU).*
3. **Generate Figures:** Both notebooks will populate the `figures/` directory with the exact plots used in our final presentation.

*Note: Please be aware of custom Paths and change them accordingly in your notebooks! Also adapt the Batchsize when training to fit your Hardware!*

## 4. Author contributions
* **Harris** implemented the NeuroGym task wrappers, continuous-time RNN architecture, and researched the networks hiddensize influence.
* **Jean-Pasqual** finished the `data_generator.py` pipeline and CTRNN architecture, configured the training pipeline, and conducted the Elman vs. CTRNN performance comparison.
* **Jan** adapted the training pipeline, wrote the Partial Information Decomposition (PID) code, and analyzed and generated the final information-geometry figures.
* All authors contributed to the literature review, project planning, experimental design, drafting of the `technical_note.md`, and preparation of the final presentation slides.

## 5. Documentation of LLM Usage
We used Claude Code **VERSION**, Github Copilot **VERSION**, **ChatGPT???** , and Gemini 1.5 Pro to assist in writing and debugging training and PID pipelines, troubleshooting compatibility issues with NeuroGym environments, generating boilerplate plotting code for matplotlib, and commenting scripts. All theoretical interpretations, mathematical derivations of the PID targets, and final code structuring were driven by the authors or based on references.
