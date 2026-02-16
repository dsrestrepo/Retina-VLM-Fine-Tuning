import os
import torch
import argparse
import re
import logging
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoProcessor
from trl import GRPOConfig, GRPOTrainer
from src.data_retina import RetinaDataset
from src.evaluate_retina import get_model_class
import src.prompts_retina as prompts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data_generator(dataset, dataset_name, tasks_list, strategy="cot"):
    """Yields samples for RLVR training."""
    
    gt_map = {
        "referable_dr": "Task_Referable",
        "binary_dr": "DR_2_Class",
    }

    for i in range(len(dataset)):
        row = dataset.get_row(i)
        img = dataset.get_image(i)
        if img is None: continue
        
        for task in tasks_list:
            gt_col = gt_map.get(task)
            if not gt_col: continue

            gt_val = row.get(gt_col, -1) 
            response = None
            
            # Check if valid binary label exists
            if gt_val == 1: response = "yes"
            elif gt_val == 0: response = "no"

            if response is None:
                continue

            prompt_func = prompts.get_prompt_func(dataset_name, task, strategy)
            if not prompt_func: continue
            
            text_input = prompt_func(row)
            
            # Construct the prompt structure
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text_input}
                    ]
                }
            ]
            
            yield {
                "prompt": conversation,
                "answer": response,
                "image": img 
            }

# Reward Functions
def correctness_reward_func(prompts, completions, answer, **kwargs):
    """
    Reward function that checks if the completion contains the correct answer.
    """
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        completion_lower = completion.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Check for exact match at the end or distinct word
        matches = list(re.finditer(r'\b(yes|no)\b', completion_lower))
        
        if matches:
            last_match = matches[-1].group(1)
            if last_match == ground_truth_lower:
                rewards.append(1.0)
            else:
                rewards.append(0.0) 
        else:
            rewards.append(0.0) 
            
    return rewards

def format_reward_func(prompts, completions, **kwargs):
    """
    Reward function to encourage proper formatting (length and reasoning).
    Only applies if the prompt explicitly asks for reasoning (CoT).
    """
    final_rewards = []
    reasoning_keywords = ["because", "due to", "indicates", "therefore", "suggests"]
    
    # Keywords that indicate a CoT prompt
    cot_indicators = ["reasoning", "step by step", "explain", "why"]

    for prompt, completion in zip(prompts, completions):
        # 1. Check if the prompt asks for reasoning (Automatic Detection)
        # We check the last part of the prompt (the user query) usually
        is_cot = any(indicator in prompt.lower() for indicator in cot_indicators)
        
        if not is_cot:
            # If not CoT, we want to encourage concise answers (preferably "yes" or "no")
            if completion.strip().lower() in ["yes", "no"]:
                final_rewards.append(0.1)  # Small reward for concise correct answer
            else:
                final_rewards.append(0.0)  # No reward for non-concise answers
            continue 

        # 2. Apply CoT rewards
        score = 0.0
        words = completion.split()
        
        # Encourage length for CoT
        if len(words) >= 20: 
            score += 0.1
        
        # Encourage reasoning keywords for CoT
        if any(keyword in completion.lower() for keyword in reasoning_keywords):
            score += 0.1
            
        final_rewards.append(score)
        
    return final_rewards

def main():
    parser = argparse.ArgumentParser(description="RLVF (GRPO) for MedGemma")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="google/medgemma-4b-it") 
    parser.add_argument("--output_dir", type=str, default="results/rlvf_medgemma")
    
    # GRPO specific args
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Data args
    parser.add_argument("--dataset_name", type=str, default="both", choices=["brset", "mbrset", "both"])
    parser.add_argument("--task", type=str, default="referable_dr", choices=["referable_dr", "binary_dr"])
    parser.add_argument("--prompt_strategy", type=str, default="cot", choices=["base", "cot", "role"])

    args = parser.parse_args()
    
    logger.info(f"Loading processor and model: {args.model_id}")

    # Use models.py class to load
    ModelClass = get_model_class(args.model_id)
    logger.info(f"Using model class: {ModelClass.__name__}")
    
    device_map = "auto"
    if os.environ.get("LOCAL_RANK") is not None:
        device_map = {"": int(os.environ.get("LOCAL_RANK"))}

    vlm_wrapper = ModelClass(
        model_id=args.model_id,
        quantization="4b",
        device="cuda",
        use_flash_attention=True,
        device_map=device_map
    )
    
    model = vlm_wrapper.model
    processor = vlm_wrapper.processor

    # Enable gradients for LoRA
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # LoRA Config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        task_type="CAUSAL_LM", 
        bias="none",
        lora_dropout=0.05,
    )
    
    # Load Datasets
    logger.info("Loading Datasets...")
    ds_list = []
    
    names_to_load = ["brset", "mbrset"] if args.dataset_name == "both" else [args.dataset_name]
    
    for d_name in names_to_load:
        try:
           d_set = RetinaDataset(args.dataset_path, d_name, split="train")
           logger.info(f"Loaded {d_name}: {len(d_set)} samples")
           ds_list.append((d_set, d_name))
        except Exception as e:
           logger.error(f"Skipping {d_name} due to error: {e}")

    TASKS = [args.task]

    def gen():
        for d_set, d_name in ds_list:
            yield from get_data_generator(d_set, d_name, TASKS, strategy=args.prompt_strategy)
        
    train_dataset = Dataset.from_generator(gen)
    
    # GRPO Config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none", 
        remove_unused_columns=False, 
        use_vllm=False, 
        beta=args.beta,
    )
    
    def collate_fn(features):
        """
        Custom collator to handle VLM inputs (text + images) for GRPO.
        """
        prompts = [x['prompt'] for x in features]
        answers = [x['answer'] for x in features]
        images = [x['image'] for x in features]
        
        # Apply chat template for each prompt to get the text string
        texts = [processor.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in prompts]
        
        # Prepare batched images (list of lists pattern for Gemma)
        batched_images = [[img] for img in images]
        
        # Tokenize and process images
        inputs = processor(
            text=texts,
            images=batched_images,
            padding=True,
            return_tensors="pt"
        )
        
        # Add required fields for GRPOTrainer and Reward Functions
        inputs['prompt'] = texts 
        inputs['answer'] = answers
        
        return inputs
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward_func, format_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn
    )
    
    logger.info("Starting RLVF training...")
    trainer.train()
    
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
