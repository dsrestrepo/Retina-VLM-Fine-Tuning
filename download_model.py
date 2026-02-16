# download_models.py
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# Load .env
load_dotenv()

token = os.getenv("HF_TOKEN")
if not token:
    print("Warning: HF_TOKEN not found in environment variables. Some models may require authentication.")

# List of models to download
models_to_download = [
    "Qwen/Qwen2-VL-7B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "llava-hf/llama3-llava-next-8b-hf",
    "google/medgemma-4b-it",
    "google/medgemma-1.5-4b-it",
    "Qwen/Qwen3-VL-8B-Instruct",
    "openai/gpt-oss-20b",
    # Kernel repositories required for GPT-OSS with use_kernels=True
    "kernels-community/liger_kernels",
    "kernels-community/megablocks",
    "kernels-community/triton_kernels"
]

print(f"Starting download for {len(models_to_download)} models...")

for repo_id in models_to_download:
    print(f"Downloading {repo_id}...")
    try:
        # ignore_patterns=["*.msgpack", "*.h5", "*.ot"] can save space if you only use safetensors
        path = snapshot_download(repo_id=repo_id, token=token, repo_type="model")
        print(f"Successfully downloaded {repo_id} to {path}")
    except Exception as e:
        print(f"Failed to download {repo_id}. Error: {e}", file=sys.stderr)

print("All download tasks completed.")

