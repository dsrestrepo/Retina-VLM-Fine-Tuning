#!/bin/bash
#SBATCH --job-name=rlvf_medgemma
#SBATCH --account=dtn@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=18:00:00
#SBATCH --output=logs/ft_rlvf_medgemma.out
#SBATCH --error=logs/ft_rlvf_medgemma.err

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

echo "Starting RLVF (GRPO) Training Job on Single H100..."

# -----------------------------------------------------------------------------
# 1. RLVF - REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS
# -----------------------------------------------------------------------------
echo "Running RLVF (GRPO)..."

python src/train_rlvf_medgemma.py \
    --dataset_path "$DATASET_PATH" \
    --model_id "$MODEL_ID" \
    --output_dir "$OUTPUT_BASE/rlvf_medgemma" \
    --dataset_name "both" \
    --task "referable_dr" \
    --prompt_strategy "cot" \
    --max_steps 2000 \
    --num_generations 4 \
    --batch_size 3 \
    --grad_accum 8 \
    --lora_r 16 \
    --lr 2e-6

echo "Training Job Completed."
