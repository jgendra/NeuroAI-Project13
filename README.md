# Information Decomposition in Task-Trained RNNs

![status](https://img.shields.io/badge/status-in%20progress-yellow)
![python](https://img.shields.io/badge/python-3.11-blue)
![framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)


## 1. Project summary
This project tests whether cognitive integration demands shape how information is distributed within recurrent neural networks. Closely replicating the experimental design of Mante et al. (2013), we train continuous-time RNNs (CTRNNs) on two NeuroGym tasks with differing integration requirements (Perceptual vs. Context). We evaluate the networks' internal representations using Partial Information Decomposition (PID) to quantify redundant, unique, and synergistic information. Our analysis compares these information profiles to understand how normative task pressures alter representational geometry.

## 2. Repository structure
```text
NeuroAI-Project13/
├── README.md                       # Project overview and instructions
├── technical_note.md               # Scientific motivation, hypotheses, detailed methodology, and limitations
├── environment.yml                 # Pinned conda environment
├── requirements.txt                # Pip dependencies
├── pyproject.toml                  # Project module configuration
├── notebooks/                      # Main codebase for execution and analysis
│   ├── 01_neurogym_datasets.ipynb      # Generates NeuroGym splits and stimulus coherences
│   ├── 02_train_pipeline.ipynb         # CTRNN training loops and performance tracking
│   ├── 03_PID_analysis_pipeline.ipynb  # PID computation, statistical analysis, and results figures
│   └── 04_PID_bipartitions.ipynb       # Exploring number of random bipartition splits for PID
├── src/                            # Helper modules, model definitions, and task wrappers
│   ├── analysis/                   # Gaussian MMI-PID definition and algorithms
│   ├── models/                     # Model definitions
│   ├── tasks/                      # NeuroGym task wrappers and data generation pipeline
├── results/                 
│   ├── stimulus_coherences/        # Manually saved coherence arrays for PID analysis
│   ├── model_weights/              # Saved .pt checkpoint files per seed
│   ├── model_activations/          # Hidden state tensors per seed/trial
│   ├── accuracies_n_losses/        # Training losses and accuracy logs
│   ├── pid_outputs/                # Saved PID atom arrays (Redundancy, Synergy, Unique)
|   └── unit_PID_outputs/           # Saved unit-to-all PID atom arrays (Redundancy, Synergy, Unique)
├── figures/                        # Generated plots for the presentation and technical note
├── elman_vs_ctrnn_comparison/      # Architectural comparison experiments
└── size_comparison/                # Hidden-size capacity experiments
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

1. **Dataset generation:** 
    - The datasets have to be generated to use the notebooks. They are pre-configured in `src/tasks/mante_config.py` and will be saved into the relative `data/[task]` directory.
    - Notebook `notebooks/01_neurogym_datasets.ipynb` walks through the dataset generation and shows what the actual data looks like- 
    - Alternative to reproduce our results run `python src/tasks/data_generator.py --mode context` and `python src/tasks/data_generator.py --mode perceptual`. 
    - If you use the provided model weights (without retraining), you can omit the train and validation set generation by providing the extra parameters `--n_train 0 --n_val 0`. 
    - For further details, have a look at `src/tasks/README.md`.
    - Outputs: used configuration, train, val, and by default 10 different test sets to `data/`
2. **Train the models:** 
    - Run the full sweep (20 RNNs) via notebook `notebooks/02_train_pipeline.ipynb`. The weights will be saved to `results/model_weights/[task]` and are already delivered as pre-computed. 
    - Outputs: losses and accuracies to `results/accuracies_n_losses/`, weights to `results/model_weights/`, activations to `results/model_activations/`
    - *(Expected runtime: ~2 hours on a standard GPU).*
3. **Compute PID:** 
    - Extract information metrics via notebook `notebooks/03_PID_analysis_pipeline.ipynb`. 
    - Outputs: PID metrics to `results/pid_outputs`
    - *(Expected runtime: ~10 minute on CPU, <2 minute on GPU).*
4. **Generate Figures:** 
    - Notebooks `notebooks/02_train_pipeline.ipynb` and `notebooks/03_PID_analysis_pipeline.ipynb` will populate the `figures/` directory with the exact plots used in our final presentation, together with a summary of all statistical test results `stats_summary.csv`.

**Additional Experiments:**

5. **Architecture Comparison (Elman RNN vs CTRNN):**
    - Notebook `elman_vs_ctrnn_comparison/architecture_comparison.ipynb` walks you through the comparison of both model architectures on both NeuroGym tasks.
    - Additional information and discussion on the results can be found in `technical_note.md`.
    - Outputs: 4 models, loss histories/plot, and PID plot will be saved to the respective subfolders.
6. **Hidden Size Comparison:**
    - Notebook `size_comparison/comparison.ipynb` analyzes the influence of the CTRNNs hidden size on the accuracy and PID information geometry.
    - Detailed explanation can be found in `size_comparison/README.md` and discussion of the results can be found in `technical_note.md`.
    - Outputs: loss/accuracy metrics, models, and PID results into the `size_comparison/results` folder.
7. **Number of Bipartitions for PID:**
    - Notebook `notebooks/04_PID_bipartitions.ipynb` studies the number of random bipartitions in the Gaussian PID calculation which yields a good tradeoff between computation speed and standard error.
    - Further details about this can be found in `technical_note.md`.
    - Outputs: creates plots of the 4 PID metrics (Redundancy, Synergy, Uniqueness 1 and 2) with deviation compared to the number of bipartitions

*Note: Please be aware of custom Paths and change them accordingly in your notebooks! Also, adapt the batch size when training to fit your Hardware!*

## 4. Author contributions
* **Harris** implemented the NeuroGym task wrappers, continuous-time RNN architecture, and researched the networks hiddensize influence.
* **Jean-Pasqual** finished the data generation pipeline and CTRNN architecture, configured the training pipeline, and conducted the Elman vs. CTRNN performance comparison.
* **Jan** adapted the training pipeline, wrote the Partial Information Decomposition (PID) algorithms, conducted the information-theoretic and statistical analysis, and generated the final information-geometry figures.
* All authors contributed to the literature review, project planning, experimental design, drafting of the technical note, and preparation of the final presentation slides.

| File / Directory | Description | Main Contributor(s) | Assistant(s) |
| :--- | :--- | :--- | :--- |
| `notebooks/01_neurogym_datasets.ipynb`, `src/tasks/` | NeuroGym environment wrappers, data generation & coherence tracking | JP, Harris | - |
| `notebooks/02_train_pipeline.ipynb` | CTRNN training loop and performance logging | Jan, JP | - |
| `notebooks/03_PID_analysis_pipeline.ipynb`, `notebooks/04_PID_bipartitions.ipynb`, `src/analysis/` | PID calculation, analysis & figures | Jan | - |
| `elman_vs_ctrnn_comparison/*` | Architectural comparison experiments | JP | - |
| `size_comparison/*` | Hidden size capacity experiments | Harris | - |
| `src/models/` | Elman RNN/CTRNN class definitions | JP | Harris, Jan |
| `technical_note.md` | Scientific write-up and methodology | All | - |

## 5. Documentation of LLM Usage
We used Claude Opus 4.8 / Sonnet 5, GitHub Copilot, ChatGPT 5.5, and Gemini 3.1 Pro / 3.5 Flash to assist in writing and debugging training and PID pipelines, troubleshooting compatibility issues with NeuroGym environments, generating boilerplate plotting code for matplotlib, and commenting scripts. All theoretical interpretations, mathematical derivations of the PID targets, and final code structuring were driven by the authors or based on references.
