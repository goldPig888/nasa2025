# nasa2025

This repository contains tools and scripts for my nasa2025 project.

## Environment setup (macOS / zsh)

Prerequisites
- Install Miniconda or Anaconda (https://docs.conda.io)
- Optional: VS Code with the Python extension

1. Create environment (from `environment.yml`)

```bash
# from repo root
conda env create -f environment.yml
conda activate nasa2025
```

2. Install (or re-install) Python packages from `requirements.txt` 

```bash
# with the environment active
pip install -r requirements.txt
```

