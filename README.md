# Retina LLM Fine Tuning

A comprehensive framework for fine-tuning and evaluating Large Language Models (LLMs) and Vision-Language Models (VLMs) on retinal imaging tasks. This project implements pipelines for Supervised Fine-Tuning (SFT) and Reinforcement Learning (RLVF/GRPO) to enhance the diagnosis and analysis of Diabetic Retinopathy, benchmarking models like MedGemma, Llama 3, and Qwen-VL.

## Overview

This repository contains code and scripts designed for:
- **Evaluation**: Benchmarking various open (Llama 3, Qwen-VL, MedGemma) and closed source models on Diabetic Retinopathy datasets (BRSET, MBRSET).
- **Fine-Tuning**:
  - Supervised Fine-Tuning (SFT) pipeline.
  - Reinforcement Learning with Verifiable Rewards (RLVF) using GRPO.
- **Analysis**: Tools for extracting and analyzing performance metrics.

## Project Structure

- **`src/`**: Core Python source code for training, evaluation, and data processing.
- **`scripts/`**: (Root directory) Bash scripts for submitting jobs tasks (SLURM support included).
  - `evaluate_*.sh`: Scripts for running model evaluations.
  - `ft_*.sh`: Scripts for fine-tuning (SFT and RLVF).
  - `run_*.sh`: Helper scripts for running specific model types.
- **`src_ref/`**: Reference notebooks and materials.

## Setup & Installation

1.  **Environment**: The project includes `requirements.txt` for Python dependencies.
2.  **Jean Zay Users**: The `install_packages.sh` script is provided to set up the environment on the Jean Zay supercomputer.

```bash
pip install -r requirements.txt
```

## Usage

### Fine-Tuning
Scripts are provided for both SFT and RLVF approaches:
- `ft_sft_medgemma.sh`: Run Supervised Fine-Tuning.
- `ft_rlvf_medgemma.sh`: Run RLVF with GRPO.

### Evaluation
Evaluations can be run using the provided bash scripts, which handle SLURM job submission:
- `evaluate_retina.sh`: Main evaluation script.
- `evaluate_open_llms.py` / `evaluate_open_vlms.py`: Python entry points for evaluation logic.

## Contact

For questions or inquiries, please contact:
**David Restrepo** - dsrestrepo2@gmail.com
