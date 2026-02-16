#!/bin/bash
#SBATCH --job-name=eval_retina
#SBATCH --account=dtn@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/eval_retina.out
#SBATCH --error=logs/eval_retina.err

# Clean modules
module purge
module load arch/h100
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

# Add project root to PYTHONPATH to ensure src module can be imported
export PYTHONPATH=$PYTHONPATH:.

# Define Paths
DATA_PATH="/lustre/fswork/projects/rech/dtn/uap38od/datasets"
OUTPUT_DIR="results/evals"

# Models List (Local)
# Note: GPT-5 is excluded as it requires Internet access (API), which is typically not available 
# on H100 compute nodes in offline mode.

MODELS=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "google/medgemma-4b-it"
    "google/medgemma-1.5-4b-it"
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
    "llava-hf/llama3-llava-next-8b-hf"
    "google/medgemma-27b-it" 
)

# Tasks and Datasets
#TASKS=("binary_dr" "referable_dr")
TASKS=("binary_dr")
#DATASETS=("brset" "mbrset")
DATASETS=("mbrset")
#STRATEGIES=("base" "cot" "role")
STRATEGIES=("base")

echo "Starting Evaluation Job on H100..."
echo "Models: ${MODELS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Strategies: ${STRATEGIES[*]}"

for dataset in "${DATASETS[@]}"; do
    for task in "${TASKS[@]}"; do
        for strategy in "${STRATEGIES[@]}"; do
            for model in "${MODELS[@]}"; do
                echo "----------------------------------------------------------------"
                echo "Running Evaluation: $model | Dataset: $dataset | Task: $task | Strategy: $strategy"
                echo "----------------------------------------------------------------"
                
                # Use 'time' to track duration
                time python src/evaluate_retina.py \
                    --dataset_path "$DATA_PATH" \
                    --dataset_name "$dataset" \
                    --task "$task" \
                    --model_id "$model" \
                    --prompt_strategy "$strategy" \
                    --output_dir "$OUTPUT_DIR" \
                    --batch_size 16 \
                    --split "test" \
                    --quantization "4b" \
                    --use_flash_attn || echo "Error running $model on $dataset/$task/$strategy"
                
                echo "----------------------------------------------------------------"
            done
        done
    done
done

echo "Job completed."
