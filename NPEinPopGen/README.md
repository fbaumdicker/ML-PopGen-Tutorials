Welcome to the workshop of applying Neural Posterior Estimation in inferring historical population sizes!

Before we begin, ensure the following:
1. **Operating System**: Linux/macOS/Windows (with WSL or an equivalent environment).
2. **Hardware**:
    - Only CPU is needed for this workshop.
    - [ ] GPU usage will be provided in another Notebook.
3. **Software**:
    - Python 3.9+ [sbi0.22.0](https://github.com/sbi-dev/sbi/releases/tag/v0.22.0).
    - [conda](https://docs.conda.io/en/latest/) (or `venv`) for environment management.
    - Required Python libraries for this tutorial ([requirements](https://github.com/fbaumdicker/ML-PopGen-Tutorials/blob/main/NPEinPopGen/popgensbi.yml)).

#### Environment Setup

To run this notebook, please follow these steps:
1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) if you haven’t already.
2. Clone the repository: `git clone https://github.com/fbaumdicker/ML-PopGen-Tutorials.git` and navigate to the NPEinPopGen directory.
3. Create the environment: `conda env create -f popgensbi.yml`
4. Activate the environment: `conda activate popgensbi`
5. Start notebook kernel: `python -m ipykernel install --user --name popgensbi --display-name "popgensbi"`
6. Launch Jupyter notebook: `jupyter notebook`.
7. In the Notebook, select the "popgensbi" kernel if prompted.

To run the Snakemake pipeline, please adjust the parameters in the config, and use the following command:
```
snakemake --use-conda --cores 4
```
