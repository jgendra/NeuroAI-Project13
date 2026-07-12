# Information Decomposition in Task-Trained RNNs

![status](https://img.shields.io/badge/status-in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.11-blue)
![framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)


## 1. Project summary
This project tests whether cognitive integration demands shape how information is distributed within recurrent neural networks. Closely replicating the experimental design of Mante et al. (2013), we train continuous-time RNNs (CTRNNs) on two NeuroGym tasks with differing integration requirements (Perceptual vs. Context). We evaluate the networks' internal representations using Partial Information Decomposition (PID) to quantify redundant, unique, and synergistic information. Our analysis compares these information profiles to understand how normative task pressures alter representational geometry.

## 2. Repository structure
```text
NeuroAI-Project13/
├── README.md                # Project overview and instructions
├── technical_note.md        # Scientific motivation, hypotheses, detailed methodology, and limitations
├── environment.yml          # Pinned conda environment
├── requirements.txt         # Pip dependencies
├── pyproject.toml           # Project module configuration
├── notebooks/               # Main codebase for execution and analysis
│   ├── 01_dataset_generation.ipynb   # Generates NeuroGym splits and stimulus coherences
│   ├── 02_model_training.ipynb       # CTRNN training loops and performance tracking
│   ├── 03_pid_analysis.ipynb         # Information decomposition (MMI-PID) algorithms
│   ├── ctrnn_vs_elman.ipynb          # Architectural comparison experiments
│   └── size_comparison.ipynb         # Hidden-size capacity experiments
├── src/                     # Helper modules, model definitions, and task wrappers
│   ├── analysis/            # Gaussian PID analysis, tests, and figure generation
│   ├── models/              # Model definitions
│   ├── tasks/               # NeuroGym task wrappers and data generation pipeline
├── results/                 
│   ├── stimulus_coherences/ # Manually saved coherence arrays for PID analysis
│   ├── model_weights/       # Saved .pt checkpoint files per seed
│   ├── model_activations/   # Hidden state tensors per seed/trial
│   ├── metrics/             # Training losses and accuracy logs
│   └── pid_outputs/         # Saved PID atom arrays (Redundancy, Synergy, Unique)
└── figures/                 # Generated plots for the presentation and technical note
```

Directory Details:
- `notebooks/`: Contains the primary execution code for the project. These notebooks drive data generation, training, and analysis.
- `src/`: Contains reusable Python modules imported by the notebooks.
- `results/`: The central storage hub for all generated data. Separated into subdirectories for stimuli, model checkpoints, raw activations, performance metrics, and final PID values.

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

0. **Dataset generation:** 
    - The datasets have to be generated to use the notebooks. They are pre-configured in `src/tasks/mante_config.py` and will be saved into the relative `data/[task]` directory. 
    - To reproduce our results run `python src/tasks/data_generator.py --mode context` and `python src/tasks/data_generator.py --mode perceptual`. 
    - If you use the provided model weights (without retraining) one can ommit the train and validation set generation by providing the extra parameters `--n_train 0 --n_val 0`. For further details have a look at `src/tasks/README_data.md`.
    - Outputs: used configuration, train, val, and by default 10 different test sets to `data/`
1. **Train the models:** 
    - Run the full sweep (20 RNNs) via notebook `notebooks/05_Full_train_pipeline.ipynb`. The weights will be saved to `results/model_weights/[task]` and are already delivered as pre-computed. 
    - Outputs: losses and accuracies to `results/accuracies_n_losses/`, weights to `results/model_weights/`, activations to `results/model_activations/`
    - *(Expected runtime: ~2 hours on a standard GPU).*
2. **Compute PID:** 
    - Extract information metrics via notebook `notebooks/06_Full_PID_analysis_pipeline.ipynb`. 
    - Outputs: PID metrics to `results/pid_outputs`
    - *(Expected runtime: ~10 minute on CPU, <1 minute on GPU).*
3. **Generate Figures:** 
    - Notebooks `notebooks/05_Full_train_pipeline.ipynb` and `notebooks/06_Full_PID_analysis_pipeline.ipynb` will populate the `figures/` directory with the exact plots used in our final presentation, together with a summary of all statistical test results `stats_summary.csv`.

*Note: Please be aware of custom Paths and change them accordingly in your notebooks! Also adapt the Batchsize when training to fit your Hardware!*

## 4. Author contributions
* **Harris** implemented the NeuroGym task wrappers, continuous-time RNN architecture, and researched the networks hiddensize influence.
* **Jean-Pasqual** finished the data generation pipeline and CTRNN architecture, configured the training pipeline, and conducted the Elman vs. CTRNN performance comparison.
* **Jan** adapted the training pipeline, wrote the Partial Information Decomposition (PID) code, and analyzed and generated the final information-geometry figures.
* All authors contributed to the literature review, project planning, experimental design, drafting of the technical note, and preparation of the final presentation slides.

| File / Directory | Description | Main Contributor(s) | Assistant(s) |
| :--- | :--- | :--- | :--- |
| `notebooks/01_neurogym_datasets.ipynb`, `src/tasks/` | NeuroGym environment wrappers, data generation & coherence tracking | JP, Harris | - |
| `notebooks/02_model_training_pipeline.ipynb` | CTRNN training loop and performance logging | Jan, JP | - |
| `notebooks/03_PID_analysis_pipeline.ipynb`, `src/analysis/` | MMI-PID calculation on hidden states | Jan | - |
| `elman_vs_ctrnn_comparison/` | Architectural comparison experiments | JP | - |
| `size_comparison/` | Hidden size capacity experiments | Harris | - |
| `src/models/` | Elman RNN/CTRNN class definitions | JP | Harris, Jan |
| `technical_note.md` | Scientific write-up and methodology | All | All |

## 5. Documentation of LLM Usage
We used Claude Code **VERSION**, Github Copilot **VERSION**, **ChatGPT???** , and Gemini 1.5 Pro to assist in writing and debugging training and PID pipelines, troubleshooting compatibility issues with NeuroGym environments, generating boilerplate plotting code for matplotlib, and commenting scripts. All theoretical interpretations, mathematical derivations of the PID targets, and final code structuring were driven by the authors or based on references.
