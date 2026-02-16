import os
import torch
import argparse
import re
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
from trl import GRPOConfig, GRPOTrainer
from src.data_retina import RetinaDataset
from src.evaluate_retina import get_model_class
import src.prompts_retina as prompts
import logging

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
            
            # Construct the prompt structure for GRPOTrainer
            # We follow the chat template structure
            # The system prompt can be added if needed, but MedGemma usually relies on user/model turns
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text_input}
                    ]
                }
            ]
            
            # For GRPO, we need 'prompt' (which is the conversation) and 'answer' (ground truth)
            yield {
                "prompt": conversation,
                "answer": response,
                "image": img 
                # Pass image separately? GRPOTrainer usually expects it in the prompt if using process_class
                # But let's see how we handle it.
                # Actually passing 'prompt' as a list of dicts + 'image' might not be standard for all trainers yet.
                # But if we use a value-aligned trainer, we might need a custom data collator or rely on the processor.
                # For now, let's keep 'image' in the dict and we'll handle it in the collator if needed, 
                # or hope the processor handles the 'prompt' structure if we pass it correctly.
            }

# Reward Functions
def correctness_reward_func(prompts, completions, answer, **kwargs):
    """
    Reward function that checks if the completion contains the correct answer.
    We look for the last occurrence of 'yes' or 'no' in the completion.
    """
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        # Normalize text
        completion_lower = completion.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Simple extraction: check if the answer is present as a distinct word
        # Ideally, we want the MODEL to output "Answer: yes" or something similar.
        # But if it's free form CoT, it might end with "Therefore, the answer is yes."
        
        # Check for exact match at the end or distinct word
        # Let's find all occurrences of "yes" and "no"
        matches = list(re.finditer(r'\b(yes|no)\b', completion_lower))
        
        if matches:
            # Take the last one as the final answer
            last_match = matches[-1].group(1)
            if last_match == ground_truth_lower:
                rewards.append(1.0)
            else:
                rewards.append(0.0) # Wrong answer
        else:
            rewards.append(0.0) # No answer found
            
    return rewards

def format_reward_func(prompts, completions, **kwargs):
    """
    Reward function to encourage proper formatting.
    For CoT, we want to see some reasoning length or specific keywords before the answer.
    """
    rewards = []
    for completion in completions:
        # Check length as a proxy for reasoning (CoT should be verbose-ish)
        # If it's too short, it's probably not reasoning.
        # Threshold: e.g., > 20 words for reasoning?
        
        # Or check for structure like <think> ... </think> if the model supports it.
        # But MedGemma 4B might not use explicit XML tags by default unless trained.
        # Let's just encourage length for now.
        
        words = completion.split()
        if len(words) >= 20: 
            rewards.append(0.5) # Small bonus for reasoning
        else:
            rewards.append(0.0)
            
        # Example: Check for "reasoning" or "because"
        if "because" in completion.lower() or "due to" in completion.lower() or "indicates" in completion.lower():
             rewards.append(0.5)
        else:
             rewards.append(0.0)

    # Summing up per completion (but we need to return a list of floats, one per completion)
    # So we should combine these logic into a single score per completion
    final_rewards = []
    for completion in completions:
        score = 0.0
        words = completion.split()
        if len(words) >= 20: 
            score += 0.1
        
        # Encourage reasoning keywords
        if any(keyword in completion.lower() for keyword in ["because", "due to", "indicates", "therefore", "suggests"]):
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
    
    print(f"Loading processor and model: {args.model_id}")

    # Use models.py class to load
    ModelClass = get_model_class(args.model_id)
    print(f"Using model class: {ModelClass.__name__}")
    
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
    
    # Prepare Model (PEFT) happens inside GRPOTrainer or we can do it here
    # GRPOTrainer usually handles PEFT config if passed.
    
    # Load Datasets
    print(f"Loading Datasets...")
    ds_list = []
    
    # If "both", we load both. Else load specific.
    names_to_load = ["brset", "mbrset"] if args.dataset_name == "both" else [args.dataset_name]
    
    for d_name in names_to_load:
        try:
           d_set = RetinaDataset(args.dataset_path, d_name, split="train")
           print(f"Loaded {d_name}: {len(d_set)} samples")
           ds_list.append((d_set, d_name))
        except Exception as e:
           print(f"Skipping {d_name} due to error: {e}")

    # Define tasks
    TASKS = [args.task]

    def gen():
        for d_set, d_name in ds_list:
            yield from get_data_generator(d_set, d_name, TASKS, strategy=args.prompt_strategy)
        
    train_dataset = Dataset.from_generator(gen)
    
    # Filter out None images if any (though generator handles it)
    
    # GRPO Config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=2048, # VLM prompts can be long due to image tokens? 
        max_completion_length=512,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none", # or "tensorboard"
        remove_unused_columns=False, # Important for custom columns like 'image'
        use_vllm=False, # We might not be able to use vLLM easily with custom VLM inputs inside GRPOTrainer yet without correct setup
        beta=args.beta,
    )
    
    # Custom Data Collator
    # GRPOTrainer expects 'prompt' to be tokenized or text.
    # If we pass processing_class, it will try to process 'prompt'.
    # But we have images.
    # We need to integrate images into the batch.
    
    # Since GRPOTrainer's default collator might not handle our 'image' column correctly combined with 'prompt' processing for VLM,
    # we might need to rely on the processor being called correctly.
    
    # However, GRPOTrainer (as of trl 0.12+) supports multimodal if the model/processor supports it.
    # But usually it requires the dataset to follow a specific structure or a custom collator.
    
    # Let's try to define a lightweight collator that uses the processor.
    # But GRPOTrainer does generation.
    
    # IMPORTANT: The 'prompt' in the dataset is a list of dicts (chat).
    # The processor needs to be called on this.
    
    # Let's try to let GRPOTrainer handle it by passing processing_class=processor.
    # And ensuring our dataset has 'prompt' (which it does) and 'completion' (which we don't need for training, but we have 'answer' for reward).
    # But the 'image' field needs to be picked up.
    
    # If GRPOTrainer doesn't support 'image' column automatically, we need to hack it.
    # One way is to pre-process the dataset to have 'pixel_values' or 'image_features' if possible,
    # but GRPOTrainer needs to generate, so it needs inputs formatted for generation.
    
    # Let's assume we pass the processor and it handles the "prompt" list which contains {"type": "image"}.
    # BUT, the PIL image object usually needs to be passed separately or somehow referenced.
    # In standard transformers chat template for VLMs, you often pass images=... to processor.
    
    # Hack: We can define a collator that processes the prompts + images into inputs.
    # GRPOTrainer will use these inputs to generate.
    
    def collate_fn(features):
        prompts = [x['prompt'] for x in features]
        answers = [x['answer'] for x in features]
        images = [x['image'] for x in features]
        
        # Process inputs for generation
        # Gemma 3 processor expects images as list of lists for batching if following previous pattern?
        # Or just list of images if matching prompts.
        # Let's follow the pattern we found successful in SFT: list of lists for batched?
        # Wait, in SFT we did processed_images.append([ex['image']]).
        # Here we have a batch of features.
        
        # Prepare valid inputs for model.generate
        # We need to apply chat template to get text, and pass images.
        
        # Apply chat template for each prompt to get the text string
        texts = [processor.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in prompts]
        
        # Prepare images: list of lists?
        # If batch size is B, images arg should be [[img1_seq], [img2_seq], ...] ?
        # Yes, based on SFT experience.
        batched_images = [[img] for img in images]
        
        # Tokenize
        inputs = processor(
            text=texts,
            images=batched_images,
            padding=True,
            return_tensors="pt"
        )
        
        # GRPOTrainer expects the collator to return a batch that can be passed to model.generate
        # The batch should also contain 'prompt' (text) and 'answer' (for reward function).
        # We can add them to the inputs dict (which is what usually happens).
        # But 'prompt' and 'answer' are not tensors, so we must be careful if the trainer expects consistent types.
        # Usually GRPOTrainer handles non-tensor keys if they are in the dataset, but since we are writing a collator, we must duplicate them?
        
        # Re-reading GRPOTrainer docs/source logic (via knowledge training):
        # The trainer expects the dataset to have 'prompt'.
        # If we provide a collator, we must return what the model needs + 'prompt' + 'completion' (if available) + 'answer' (if using it in reward).
        
        # Ideally, we return the inputs dict (input_ids, pixel_values etc)
        # AND we add 'prompt' and 'answer' to it so they are passed to the loop and then to evaluation?
        # Actually GRPOTrainer has specific handling.
        
        # If we use `processing_class`, GRPOTrainer might try to collate automatically.
        # But automatic collation fails for images usually.
        
        # So manual collator is safer.
        
        # Add metadata for rewards
        # We pass the original prompts structure (list of dicts) or the string?
        # Reward functions receive 'prompts' and 'completions'.
        # 'prompts' will be what we return here under key 'prompt' or 'prompts'?
        # GRPOTrainer looks for 'prompt' or 'prompts' in the batch.
        
        inputs['prompt'] = texts # Pass the text string as prompt for reference / reward? 
                                 # Or pass the original structure? 
                                 # Correctness reward just needs 'answer'.
                                 # Format reward checks 'completion'.
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
    
    print("Starting RLVR training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
