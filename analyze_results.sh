#!/bin/bash
#SBATCH --job-name=analyze_retina_metrics
#SBATCH --account=dtn@a100
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/analyze_metrics.out
#SBATCH --error=logs/analyze_metrics.err

# Clean modules
module purge
module load arch/a100
# Load required modules
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

# Define Paths
EVAL_DIR="results/evals"
METRICS_DIR="results/metrics"
EXTRACTOR_MODEL="google/medgemma-27b-it" 
QUANTIZATION="4b"

echo "Starting Metrics Analysis Job on GPU (A100)..."
echo "Evaluations Directory: $EVAL_DIR"
echo "Metrics Directory: $METRICS_DIR"
echo "Extractor Model: $EXTRACTOR_MODEL"

# Install report libraries if missing
#pip install scikit-learn matplotlib seaborn || echo "Libraries already installed"

# Run Analysis Script
python src/analyze_metrics.py \
    --eval_dir "$EVAL_DIR" \
    --metrics_dir "$METRICS_DIR" \
    --extractor_model_id "$EXTRACTOR_MODEL" \
    --quantization "$QUANTIZATION" \
    --device "cuda"

echo "Analysis Job Completed."
