#!/bin/bash
#SBATCH --job-name=train_retllm
#SBATCH --account=dtn@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=18:00:00
#SBATCH --output=logs/ft_sft_medgemma.out
#SBATCH --error=logs/ft_sft_medgemma.err

# Clean modules
module purge
module load arch/h100
module load miniforge/24.9.0
module load gcc/11.4.1
module load cuda/12.4.1
module load cudnn/9.2.0.82-cuda
module load nccl/2.21.5-1-cuda

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate llms

# Environment Configuration
export HF_HOME="$WORK/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:.

# Configuration
DATASET_PATH="$WORK/datasets" 
MODEL_ID="google/medgemma-4b-it" 
OUTPUT_BASE="results/checkpoints"


echo "Starting Training Job on Single H100..."

# -----------------------------------------------------------------------------
# 1. Supervised Fine-Tuning (SFT)
# -----------------------------------------------------------------------------
echo "Running SFT Training..."

python src/train_sft_medgemma.py \
    --dataset_path "$DATASET_PATH" \
    --model_id "$MODEL_ID" \
    --output_dir "$OUTPUT_BASE/sft_medgemma" \
    --dataset_name "both" \
    --task "referable_dr" \
    --epochs 3 \
    --batch_size 8 \
    --grad_accum 4 \
    --lora_r 16 \
    --lr 2e-5

echo "Training Job Completed."
