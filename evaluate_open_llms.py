import os
import gc
import torch
from dotenv import load_dotenv
from src.models import GptOssLLM

load_dotenv()

def run_test(llm_instance, model_name):
    print(f"\n[{model_name}] Starting LLM evaluation...")
    
    # Test 1: Simple Question (Default Reasoning)
    print(f"[{model_name}] Test 1: Simple Question (Default Reasoning)")
    try:
        res = llm_instance.generate(prompt="What is the capital of France?", max_new_tokens=100)
        print(f"[{model_name}] Result: {res['text']}")
    except Exception as e:
        print(f"[{model_name}] FAILED 1: {e}")

    # Test 2: Reasoning Question (High Reasoning Effort)
    print(f"[{model_name}] Test 2: Reasoning Question (High Reasoning Effort)")
    try:
        # Note: reasoning_effort is passed to generate, which GptOssLLM handles
        res = llm_instance.generate(prompt="Explain why the sky is blue.", max_new_tokens=500, reasoning_effort="high")
        print(f"[{model_name}] Result: {res['text']}")
    except Exception as e:
        print(f"[{model_name}] FAILED 2: {e}")

def cleanup(llm):
    if llm: del llm
    gc.collect()
    torch.cuda.empty_cache()

def main():
    print("=== STARTING OPEN LLM EVALUATION (GPU / OFFLINE) ===")
    
    # Enable Offline Mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # List of models to test
    # (Name, Class, ID, Quantization)
    models = [
        ("GPT-OSS 20B", GptOssLLM, "openai/gpt-oss-20b", "4b"), 
        # Add other pure LLMs here if needed
    ]

    for name, cls, mid, quant in models:
        print(f"\n>>> Loading {name} ({mid}) Quant: {quant}")
        llm = None
        try:
            # device="cuda" is standard for GPU job
            llm = cls(model_id=mid, device="cuda", quantization=quant, offline_mode=True)
            run_test(llm, name)
        except Exception as e:
            print(f"FAILED LOADING {name}: {e}")
        finally:
            cleanup(llm)
    
    print("\nEvaluation Done.")

if __name__ == "__main__":
    main()
