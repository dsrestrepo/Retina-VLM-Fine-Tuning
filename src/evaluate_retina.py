import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Import models
from src.models import (
    QwenVLM, LlavaVLM, LlamaVLM, GemmaVLM,
    OpenAIVLM, GeminiVLM, GptOssLLM
)
from src.data_retina import RetinaDataset
import src.prompts_retina as prompts

def get_model_class(model_id):
    model_id_lower = model_id.lower()
    if "qwen" in model_id_lower: return QwenVLM
    if "llava" in model_id_lower: return LlavaVLM
    if "llama" in model_id_lower and "vision" in model_id_lower: return LlamaVLM # Llama 3.2 Vision
    if "llama" in model_id_lower: return GptOssLLM # Llama text only? Or check mllama
    if "gemma" in model_id_lower: return GemmaVLM
    if "gpt" in model_id_lower and "oss" not in model_id_lower: return OpenAIVLM
    if "gemini" in model_id_lower: return GeminiVLM
    # Fallback/Default
    if "mllama" in model_id_lower: return LlamaVLM
    return GptOssLLM

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on Retina Datasets")
    parser.add_argument("--dataset_path", type=str, required=True, help="Base path to datasets")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["brset", "mbrset"])
    parser.add_argument("--task", type=str, required=True, 
                        choices=["referable_dr", "binary_dr", "dr_5", "dr_3", "glaucoma", "amd"])
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--quantization", type=str, default=None, choices=["4b", "8b", "16b"])
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on (train, val, test, all)")
    parser.add_argument("--prompt_strategy", type=str, default="base", choices=["base", "cot", "role"], help="Prompting strategy to use")
    
    args = parser.parse_args()
    
    # 1. Load Dataset
    print(f"Loading {args.dataset_name} ({args.split} split) from {args.dataset_path}...")
    dataset = RetinaDataset(args.dataset_path, args.dataset_name, split=args.split)
    print(f"Loaded {len(dataset)} samples.")

    # 2. Config Task
    # Define task configs for each dataset
    
    # Common prompts map
    # Note: We use specific prompt functions for each dataset/task combination
    
    # BRSET Configs
    brset_tasks = {
        "referable_dr": {
            "prompt_func": prompts.BRSET_REFERABLE_DR_PROMPT,
            "gt_col": "Task_Referable"
        },
        "binary_dr": {
            "prompt_func": prompts.BRSET_BINARY_DR_PROMPT,
            "gt_col": "DR_2_Class"
        },
        "dr_5": {
            "prompt_func": prompts.BRSET_5_CLASS_DR_PROMPT,
            "gt_col": "DR_ICDR"
        },
        "dr_3": {
            "prompt_func": prompts.BRSET_3_CLASS_DR_PROMPT,
            "gt_col": "Task_3_Classes"
        },
        "glaucoma": {
            "prompt_func": prompts.BRSET_GLAUCOMA_PROMPT,
            "gt_col": "Task_Glaucoma"
        },
        "amd": {
            "prompt_func": prompts.BRSET_AMD_PROMPT,
            "gt_col": "Task_AMD"
        }
    }

    # mBRSET Configs
    mbrset_tasks = {
        "referable_dr": {
            "prompt_func": prompts.mBRSET_REFERABLE_DR_PROMPT,
            "gt_col": "Task_Referable"
        },
        "binary_dr": {
            "prompt_func": prompts.mBRSET_BINARY_DR_PROMPT,
            "gt_col": "DR_2_Class"
        },
        "dr_3": {
            "prompt_func": prompts.mBRSET_3_CLASS_DR_PROMPT,
            "gt_col": "Task_3_Classes"
        },
        # mBRSET doesn't have glaucoma/amd distinct columns in the provided data_retina.py
    }

    if args.dataset_name == "brset":
        task_config = brset_tasks
    elif args.dataset_name == "mbrset":
        task_config = mbrset_tasks
    else:
        raise ValueError(f"Unknown dataset {args.dataset_name}")
    
    config = task_config.get(args.task)
    if not config:
        raise ValueError(f"Task {args.task} is not supported for dataset {args.dataset_name} (or check logic)")

    # Select prompt function based on strategy
    # Try dynamic prompt dispatcher first
    prompt_func = prompts.get_prompt_func(args.dataset_name, args.task, args.prompt_strategy)
    
    if prompt_func is None:
        # Fallback to static config
        print(f"Strategy '{args.prompt_strategy}' not explicitly found for {args.task}, using default/base configuration.")
        prompt_func = config["prompt_func"]
    else:
        print(f"Using prompt strategy: {args.prompt_strategy}")

    gt_col = config["gt_col"]

    # 3. Load Model
    print(f"Loading model {args.model_id}...")
    ModelClass = get_model_class(args.model_id)
    
    # Handle GptOssLLM specific default like in other scripts
    quantization = args.quantization
    if ModelClass == GptOssLLM and quantization is None:
         # Default to 4b for GptOss to avoid OOM as per previous context
         quantization = "4b" 

    model = ModelClass(
        model_id=args.model_id,
        quantization=quantization,
        use_flash_attention=args.use_flash_attn
    )

    # 4. Inference Loop
    results = []
    
    # Batch indices
    indices = list(range(len(dataset)))
    
    os.makedirs(args.output_dir, exist_ok=True)
    strategy_suffix = f"_{args.prompt_strategy}" if args.prompt_strategy != "base" else ""
    output_file = os.path.join(args.output_dir, f"{args.dataset_name}_{args.task}{strategy_suffix}_{args.model_id.replace('/', '_')}.csv")

    for i in tqdm(range(0, len(indices), args.batch_size)):
        # Standard slice
        batch_indices = indices[i : i + args.batch_size]

        batch_prompts = []
        batch_images = []
        batch_rows = []

        for idx in batch_indices:
            row = dataset.get_row(idx)
            img = dataset.get_image(idx)
            
            # Construct prompt
            text_prompt = prompt_func(row)
            
            batch_prompts.append(text_prompt)
            batch_images.append(img)
            batch_rows.append(row)

        # Run Inference
        try:
            # Basic generation
            # Some models (GPT-OSS) don't support images, VLM wrapper handles it by ignoring image arg if needed
            outputs = model.generate_batch(
                prompts=batch_prompts,
                images=batch_images,
                max_new_tokens=128
            )
            
            for j, out_text in enumerate(outputs):
                res = {
                    "id": batch_rows[j].get("image_id", batch_rows[j].get("file", str(batch_indices[j]))),
                    "ground_truth": batch_rows[j].get(gt_col, -1),
                    "prediction_text": out_text,
                    "prompt": batch_prompts[j]
                }
                results.append(res)
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Save partial?
            continue
            
    # 5. Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
