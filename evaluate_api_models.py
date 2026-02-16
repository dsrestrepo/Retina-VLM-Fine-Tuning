import os
import gc
import sys
from pathlib import Path
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from src.models import OpenAIVLM, GeminiVLM

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
        ImageDraw.Draw(img).text((50,100), f"API TEST {i+1}", fill="white")
        images.append(img)
    return images

def run_test(vlm_instance, model_name):
    print(f"\n[{model_name}] Starting API evaluation...")
    images = get_test_images(n=2)
    img1 = images[0]
    img2 = images[1]
    
    # Test 1: Single Image
    print(f"[{model_name}] Test 1: Single Image")
    try:
        # Increased max_new_tokens to accommodate thinking process
        res = vlm_instance.generate(prompt="Describe this image. Be brief", image=img1, max_new_tokens=512)
        print(f"[{model_name}] Result: {res.get('text', 'No text returned')}")
    except Exception as e:
        print(f"[{model_name}] Test 1 FAILED: {e}")

    # Test 2: Multiple Images
    print(f"[{model_name}] Test 2: Multiple Images")
    try:
         res = vlm_instance.generate(prompt="What are the differences between these two images? Be brief", image=[img1, img2], max_new_tokens=512)
         print(f"[{model_name}] Result: {res.get('text', 'No text returned')}")
    except Exception as e:
        print(f"[{model_name}] Test 2 FAILED: {e}")

    # Test 3: Text Only
    print(f"[{model_name}] Test 3: Text Only")
    try:
        # Increased max_new_tokens to accommodate thinking process
        res = vlm_instance.generate(prompt="What is Diabetic Retinopathy? Be brief", image=None, max_new_tokens=512)
        print(f"[{model_name}] Result: {res.get('text', 'No text returned')}")
    except Exception as e:
        print(f"[{model_name}] Test 3 FAILED: {e}")

def main():
    print("=== STARTING API EVALUATION (CPU / ONLINE) ===")
    
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            print("\nInitializing OpenAI GPT-5...")
            # offline_mode=False is crucial here
            run_test(OpenAIVLM(model_id="gpt-5-2025-08-07", offline_mode=False), "GPT-5")
        except Exception as e:
            print(f"Error OpenAI: {e}")
    else:
        print("Skipping OpenAI: No API Key found.")

    # Gemini
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            print("\nInitializing Gemini 3 Pro...")
            run_test(GeminiVLM(model_id="gemini-3-pro-preview", offline_mode=False), "Gemini-Pro")
        except Exception as e:
            print(f"Error Gemini: {e}")
    else:
        print("Skipping Gemini: No Credentials found.")

if __name__ == "__main__":
    main()
