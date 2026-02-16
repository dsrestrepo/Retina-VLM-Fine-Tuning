import os
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from src.models import QwenVLM, LlavaVLM, LlamaVLM, GemmaVLM, GptOssLLM, OpenAIVLM

# Setup environment
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
WORK = os.environ.get('WORK', os.path.expanduser('~/JeanZay'))
MBRSET_PATH = Path(WORK) / 'datasets' / 'mBRSET' / 'mbrset'
IMAGES_PATH = MBRSET_PATH / 'images_224'
LABELS_PATH = MBRSET_PATH / 'labels_mbrset.csv'

# ============================================================================
# HELPER FUNCTIONS FOR PROMPTS
# ============================================================================

def binary_to_text(val, true_str, false_str):
    # Adjust based on data encoding (1/0, True/False, 'Yes'/'No')
    if str(val).lower() in ['1', '1.0', 'true', 'yes', 'y']:
        return true_str
    return false_str

def convert_sex_mbrset(val):
    val_str = str(val).lower()
    if val_str in ['1', 'm', 'male']:
        return 'Male'
    elif val_str in ['2', 'f', 'female']:
        return 'Female'
    return 'Unknown Sex'

education_map = {
    1: "illiterate",
    2: "with incomplete primary education",
    3: "with complete primary education",
    4: "with incomplete secondary education",
    5: "with complete secondary education",
    6: "with incomplete higher education",
    7: "with complete higher education",
    # Add loose matching if strings
    "1": "illiterate",
    "Illiterate": "illiterate"
}

# ============================================================================
# PROMPT DEFINITIONS (Copied & Adapted)
# ============================================================================

def mBRSET_TEXT_PROMPT(row):
    # Age
    age_phrase = (
        f"aged {row['age']} years" 
        if not pd.isnull(row['age']) 
        else "with age not reported"
    )

    # Diabetes duration
    diabetes_phrase = (
        f"diagnosed with diabetes for {row['dm_time']} years" 
        if not pd.isnull(row['dm_time']) 
        else "with no reported diabetes duration"
    )

    # Educational level
    # Handle potential float/int mismatch
    edu_val = row['educational_level']
    if hasattr(edu_val, 'item'): edu_val = edu_val.item()
    education = education_map.get(edu_val, "with no educational level reported")

    # Build descriptions
    sex = convert_sex_mbrset(row['sex'])
    insulin = binary_to_text(row['insulin'], "using insulin", "not using insulin")
    oral = binary_to_text(row['oraltreatment_dm'], "on oral treatment for diabetes", "not on oral treatment for diabetes")
    hypertension = binary_to_text(row['systemic_hypertension'], "with systemic hypertension", "without systemic hypertension")
    alcohol = binary_to_text(row['alcohol_consumption'], "consumes alcohol", "does not consume alcohol")
    smoking = binary_to_text(row['smoking'], "smokes", "does not smoke")
    obesity = binary_to_text(row['obesity'], "with obesity", "without obesity")
    vascular = binary_to_text(row['vascular_disease'], "has vascular disease", "does not have vascular disease")
    infarction = binary_to_text(row['acute_myocardial_infarction'], "has a history of acute myocardial infarction", "no history of acute myocardial infarction")
    nephropathy = binary_to_text(row['nephropathy'], "with nephropathy", "without nephropathy")
    neuropathy = binary_to_text(row['neuropathy'], "with neuropathy", "without neuropathy")
    diabetic_foot = binary_to_text(row['diabetic_foot'], "has diabetic foot", "does not have diabetic foot")

    # Compose patient description
    description = (
        f"A {sex} patient {age_phrase}, {diabetes_phrase}, {insulin}, and {oral}. "
        f"The patient is {hypertension}, {alcohol}, {smoking}, {obesity}, and {vascular}. "
        f"Medical history includes: {infarction}, {nephropathy}, {neuropathy}, and {diabetic_foot}. "
        f"The patient is {education}."
    )

    # LLM prompt
    return f"""
{description}

Based on the provided patient information and the associated fundus image, does the patient has Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
""".strip()

def mBRSET_ONLY_TEXT_PROMPT(row):
     # Same description logic as above, but for text-only context
    # Age
    age_phrase = (
        f"aged {row['age']} years" 
        if not pd.isnull(row['age']) 
        else "with age not reported"
    )

    # Diabetes duration
    diabetes_phrase = (
        f"diagnosed with diabetes for {row['dm_time']} years" 
        if not pd.isnull(row['dm_time']) 
        else "with no reported diabetes duration"
    )

    # Educational level
    edu_val = row['educational_level']
    if hasattr(edu_val, 'item'): edu_val = edu_val.item()
    education = education_map.get(edu_val, "with no educational level reported")

    # Build descriptions
    sex = convert_sex_mbrset(row['sex'])
    insulin = binary_to_text(row['insulin'], "using insulin", "not using insulin")
    oral = binary_to_text(row['oraltreatment_dm'], "on oral treatment for diabetes", "not on oral treatment for diabetes")
    hypertension = binary_to_text(row['systemic_hypertension'], "with systemic hypertension", "without systemic hypertension")
    alcohol = binary_to_text(row['alcohol_consumption'], "consumes alcohol", "does not consume alcohol")
    smoking = binary_to_text(row['smoking'], "smokes", "does not smoke")
    obesity = binary_to_text(row['obesity'], "with obesity", "without obesity")
    vascular = binary_to_text(row['vascular_disease'], "has vascular disease", "does not have vascular disease")
    infarction = binary_to_text(row['acute_myocardial_infarction'], "has a history of acute myocardial infarction", "no history of acute myocardial infarction")
    nephropathy = binary_to_text(row['nephropathy'], "with nephropathy", "without nephropathy")
    neuropathy = binary_to_text(row['neuropathy'], "with neuropathy", "without neuropathy")
    diabetic_foot = binary_to_text(row['diabetic_foot'], "has diabetic foot", "does not have diabetic foot")

    # Compose patient description
    description = (
        f"A {sex} patient {age_phrase}, {diabetes_phrase}, {insulin}, and {oral}. "
        f"The patient is {hypertension}, {alcohol}, {smoking}, {obesity}, and {vascular}. "
        f"Medical history includes: {infarction}, {nephropathy}, {neuropathy}, and {diabetic_foot}. "
        f"The patient is {education}."
    )

    return f"""
{description}

Based on the provided patient information, does the patient have Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
""".strip()

mBRSET_ONLY_IMAGE_TEXT_PROMPT = """
Based on the image, does the patient has Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
"""

# ============================================================================
# EVALUATION LOGIC
# ============================================================================

def evaluate_models():
    # Load Data
    if not LABELS_PATH.exists():
        print(f"Error: Labels file not found at {LABELS_PATH}")
        return
    
    df = pd.read_csv(LABELS_PATH)
    print(f"Loaded {len(df)} samples from mBRSET.")
    
    # Filter for testing? (Optional: Use full dataset)
    # Using a subset for quick testing if needed, but script is for evaluation.
    # df = df.head(10) # Comment out for full run
    
    results = []

    # Define Models
    models_config = [
        # (Name, Class, ID, Quantization, Modality)
        # Modality: "both" (Image+Text), "text" (Text Only), "image" (Image Only)
        
        # VLMs (Vision + Text)
        ("Qwen2-VL-7B", QwenVLM, "Qwen/Qwen2-VL-7B-Instruct", "4b", "both"),
        ("Llava-Next-8B", LlavaVLM, "llava-hf/llama3-llava-next-8b-hf", "4b", "both"),
        ("Llama-3.2-11B", LlamaVLM, "meta-llama/Llama-3.2-11B-Vision-Instruct", "4b", "both"),
        ("MedGemma-4B", GemmaVLM, "google/medgemma-4b-it", "4b", "both"),
        
        # LLMs (Text Only) - Using GPT-OSS
        ("GPT-OSS-20B", GptOssLLM, "openai/gpt-oss-20b", "4b", "text"),
    ]

    for model_name, ModelClass, model_id, quant, modality in models_config:
        print(f"\nExample Evaluation: {model_name} (Mode: {modality})")
        
        try:
            # Load Model
            # Note: GPT-OSS ignores '4b' internally in our class due to custom logic, which is good.
            # But passing "4b" ensures we trigger the "native MXFP4" path we coded in model.py
            model = ModelClass(model_id=model_id, quantization=quant, device="cuda", offline_mode=True)
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval {model_name}"):
                
                # Prepare Prompt & Input
                if modality == "text":
                    prompt = mBRSET_ONLY_TEXT_PROMPT(row)
                    image = None
                elif modality == "image":
                    prompt = mBRSET_ONLY_IMAGE_TEXT_PROMPT
                    img_path = IMAGES_PATH / row['file']
                    if not img_path.exists():
                        continue # Skip if image missing
                    image = Image.open(img_path).convert('RGB')
                else: # "both"
                    prompt = mBRSET_TEXT_PROMPT(row)
                    img_path = IMAGES_PATH / row['file']
                    if not img_path.exists():
                        continue
                    image = Image.open(img_path).convert('RGB')
                
                # Predict
                try:
                    res = model.generate(
                        prompt=prompt, 
                        image=image, 
                        max_new_tokens=10, # expecting yes/no
                        temperature=0.01
                    )
                    prediction_text = res['text'].strip().lower()
                except Exception as e:
                    print(f"Error generating for {row['file']}: {e}")
                    prediction_text = "error"
                
                # Ground Truth
                # ICDR >= 1 is YES (DR present)
                # ICDR == 0 is NO (No DR)
                ground_truth = "yes" if row['final_icdr'] >= 1 else "no"
                
                results.append({
                    "model": model_name,
                    "modality": modality,
                    "file": row['file'],
                    "patient_id": row['patient'],
                    "ground_truth": ground_truth,
                    "prediction_raw": prediction_text,
                    "icdr_score": row['final_icdr']
                })
            
            # Cleanup model to free VRAM for next one
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to load/run {model_name}: {e}")
            continue

    # Save Results
    if results:
        res_df = pd.DataFrame(results)
        out_file = "mbrset_evaluation_results.csv"
        res_df.to_csv(out_file, index=False)
        print(f"\nResults saved to {out_file}")
        
        # Simple Accuracy Check
        # Clean predictions: check if 'yes' or 'no' is in the text
        def clean_pred(x):
            if 'yes' in x: return 'yes'
            if 'no' in x: return 'no'
            return 'ambiguous'

        res_df['pred_clean'] = res_df['prediction_raw'].apply(clean_pred)
        res_df['correct'] = res_df['pred_clean'] == res_df['ground_truth']
        
        print("\nAccuracy per Model:")
        print(res_df.groupby('model')['correct'].mean())


if __name__ == "__main__":
    evaluate_models()


# INCLUDE
# GLAUCOMA
# AMD