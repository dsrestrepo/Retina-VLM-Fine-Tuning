import os
import gc
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from src.models import QwenVLM, LlavaVLM, LlamaVLM, GemmaVLM

load_dotenv()

def get_test_images(n=2):
    """Load real images from dataset or create dummy ones."""
    work_dir = os.environ.get('WORK', os.path.expanduser('~/JeanZay'))
    # Try BRSET first
    possible_paths = [
        Path(work_dir) / 'datasets' / 'BRSET' / 'brset' / 'images_224',
        Path(work_dir) / 'datasets' / 'mBRSET' / 'mbrset' / 'images_224'
    ]
    
    images = []
    for p in possible_paths:
        if p.exists():
            jpgs = list(p.glob("*.jpg"))
            if jpgs:
                # Get up to n images, repeat if necessary
                for i in range(n):
                    img_path = jpgs[i % len(jpgs)]
                    images.append(Image.open(img_path).convert('RGB'))
                return images
    
    # Fallback to dummy images
    colors = ['teal', 'maroon', 'navy', 'olive']
    for i in range(n):
        img = Image.new('RGB', (224, 224), color=colors[i % len(colors)])
        ImageDraw.Draw(img).text((50,100), f"IMG {i+1}", fill="white")
        images.append(img)
    return images

def run_test(vlm_instance, model_name):
    print(f"\n[{model_name}] Starting GPU evaluation...")
    images = get_test_images(n=2)
    img1 = images[0]
    img2 = images[1]
    
    # Test 1: Single Image
    print(f"[{model_name}] Test 1: Single Image")
    try:
        res = vlm_instance.generate(prompt="Describe this image.", image=img1, max_new_tokens=100)
        print(f"[{model_name}] Result: {res['text']}")
    except Exception as e:
        print(f"[{model_name}] FAILED 1: {e}")

    # Test 2: Multiple Images
    print(f"[{model_name}] Test 2: Multiple Images")
    try:
        res = vlm_instance.generate(prompt="What are the differences between these two images?", image=[img1, img2], max_new_tokens=100)
        print(f"[{model_name}] Result: {res['text']}")
    except Exception as e:
        print(f"[{model_name}] FAILED 2: {e}")

    # Test 3: Text Only
    print(f"[{model_name}] Test 3: Text Only")
    try:
        res = vlm_instance.generate(prompt="What is the capital of France?", image=None, max_new_tokens=100)
        print(f"[{model_name}] Result: {res['text']}")
    except Exception as e:
        print(f"[{model_name}] FAILED 3: {e}")

def cleanup(vlm):
    if vlm: del vlm
    gc.collect()
    torch.cuda.empty_cache()

def main():
    print("=== STARTING OPEN COMPUTE EVALUATION (GPU / OFFLINE) ===")
    
    # Enable Offline Mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # List of models to test
    # (Name, Class, ID, Quantization)
    models = [
        ("MedGemma 1.5", GemmaVLM, "google/medgemma-1.5-4b-it", "4b"), # 4b model usually
        ("MedGemma 4b", GemmaVLM, "google/medgemma-4b-it", "4b"),
        ("Qwen 2", QwenVLM, "Qwen/Qwen2-VL-7B-Instruct", "4b"),
        ("Qwen 3", QwenVLM, "Qwen/Qwen3-VL-8B-Instruct", "4b"),
        ("LLaVA Next", LlavaVLM, "llava-hf/llama3-llava-next-8b-hf", "4b"),
        ("Llama 3.2 11B", LlamaVLM, "meta-llama/Llama-3.2-11B-Vision-Instruct", "8b") # 8b fit better
    ]

    for name, cls, mid, quant in models:
        print(f"\n>>> Loading {name} ({mid}) Quant: {quant}")
        vlm = None
        try:
            # device="cuda" is standard for GPU job
            vlm = cls(model_id=mid, device="cuda", quantization=quant, offline_mode=True, use_flash_attention=True)
            run_test(vlm, name)
        except Exception as e:
            print(f"FAILED LOADING {name}: {e}")
        finally:
            cleanup(vlm)
    
    print("\nEvaluation Done.")

if __name__ == "__main__":
    main()
