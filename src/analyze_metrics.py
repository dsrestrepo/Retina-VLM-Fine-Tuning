import argparse
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    cohen_kappa_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
from tqdm import tqdm
import torch

# Import models for the extractor
from src.models import GemmaVLM

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Evaluation Metrics")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory containing evaluation CSVs")
    parser.add_argument("--metrics_dir", type=str, required=True, help="Directory to save metrics and plots")
    parser.add_argument("--extractor_model_id", type=str, default="google/medgemma-27b-it", help="Model to use for extraction if regex fails")
    parser.add_argument("--quantization", type=str, default="4b", help="Quantization for extractor model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()

class LabelExtractor:
    def __init__(self, model_id, quantization, device):
        self.model_id = model_id
        self.quantization = quantization
        self.device = device
        self.model = None
        
    def _load_model(self):
        if self.model is None:
            print(f"Loading Extractor Model: {self.model_id}...")
            self.model = GemmaVLM(
                model_id=self.model_id,
                quantization=self.quantization,
                device=self.device,
                use_flash_attention=True
            )
    
    def extract_with_regex(self, text):
        if not isinstance(text, str):
            return None
            
        text = text.strip().lower()
        
        # 1. Exact match
        if text in ["yes", "no", "yes.", "no."]:
            return 1 if "yes" in text else 0
            
        # 2. Starts with pattern
        # Matches: "yes", "yes.", "yes,", "**yes**", "answer: yes", "answer: **yes**", "answer is yes", "the answer is yes"
        # Adjusted to handle optional colon after "answer"
        start_yes_pattern = r"^(the\s+)?(answer\s*:?\s*)?(is\s+)?(\*\*)?yes(\*\*)?([.,]|$)"
        start_no_pattern = r"^(the\s+)?(answer\s*:?\s*)?(is\s+)?(\*\*)?no(\*\*)?([.,]|$)"
        
        if re.match(start_yes_pattern, text):
            return 1
        if re.match(start_no_pattern, text):
            return 0
            
        # 3. Ends with pattern
        # Matches: "..., yes.", "..., **yes**.", "..., the answer is yes."
        end_yes_pattern = r".*(^|\s)((the\s+)?(answer\s*:?\s*)?(is\s+)?(\*\*)?yes(\*\*)?)[.,]?$"
        end_no_pattern = r".*(^|\s)((the\s+)?(answer\s*:?\s*)?(is\s+)?(\*\*)?no(\*\*)?)[.,]?$"
        
        if re.search(end_yes_pattern, text):
            return 1
        if re.search(end_no_pattern, text):
            return 0
            
        return None

    def extract_with_llm(self, texts):
        if not texts:
            return []

        # Load model only if needed
        self._load_model()
        
        print(f"Running LLM extraction on {len(texts)} samples...")
        results = []
        batch_size = 8  # Reduced batch size to be safe
        
        # We need to process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="LLM Extraction"):
            batch_texts = texts[i:i+batch_size]
            batch_prompts = []
            
            for text in batch_texts:
                # Construct a prompt for the extractor model
                prompt = (
                    f"Review the following model output and determine if the final answer is 'yes' (presence of Diabetic Retinopathy) or 'no' (absence).\n"
                    f"Ignore any reasoning provided, focus only on the final conclusion.\n"
                    f"Output ONLY 'yes' or 'no'.\n\n"
                    f"Model Output: \"{text}\"\n\n"
                    f"Final Answer:"
                )
                batch_prompts.append(prompt)

            try:
                # Generate
                # Note: generate_batch expects images argument usually, passing None for text-only
                # We need to ensure the VLM class handles text-only (image=None) correctly
                outputs = self.model.generate_batch(
                    prompts=batch_prompts,
                    images=None, 
                    max_new_tokens=10,
                    temperature=0.01,
                    do_sample=False
                )
                
                for output in outputs:
                    cleaned = output.strip().lower()
                    # Check for explicit yes/no in the extractor's output
                    if "yes" in cleaned:
                        results.append(1)
                    elif "no" in cleaned:
                        results.append(0)
                    else:
                        print(f"Warning: Extractor LLM produced unclear output: '{output}'")
                        results.append(0) # Default to 0 (No DR) if unclear to avoid breaking downstream
            except Exception as e:
                print(f"Error in extractor batch: {e}")
                for _ in batch_texts:
                    results.append(0) # Fallback

        return results

def plot_confusion_matrix(y_true, y_pred, labels, output_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_args()
    
    # Ensure output directories exist
    os.makedirs(args.metrics_dir, exist_ok=True)
    plots_dir = os.path.join(args.metrics_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize Extractor
    extractor = LabelExtractor(args.extractor_model_id, args.quantization, args.device)
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(args.eval_dir, "*.csv"))
    print(f"Found {len(csv_files)} evaluation files.")
    
    global_results = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"Processing: {filename}")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Ensure required columns exist
            if 'prediction_text' not in df.columns or 'ground_truth' not in df.columns:
                print(f"Skipping {filename}: Missing 'prediction_text' or 'ground_truth'")
                continue
                
            # Filter out invalid ground truth (-1 or others if strictly binary)
            # Assuming binary tasks for now (0 vs 1)
            valid_mask = df['ground_truth'].isin([0, 1])
            if not valid_mask.all():
                print(f"Filtering {len(df) - valid_mask.sum()} rows with invalid GT from {filename}")
                df = df[valid_mask].copy()
            
            # 1. Apply Regex
            df['extracted_label'] = df['prediction_text'].apply(extractor.extract_with_regex)
            
            # 2. Identify failed extractions
            # We want to re-process ONLY the ones where regex returned None
            # Or if we want to confirm, but usually we trust the regex for clear cases.
            
            failed_indices = df[df['extracted_label'].isnull()].index
            
            if len(failed_indices) > 0:
                print(f"Regex failed for {len(failed_indices)} samples. Using LLM extractor...")
                # Get the text for these indices
                failed_texts = df.loc[failed_indices, 'prediction_text'].fillna("").astype(str).tolist()
                
                if failed_texts:
                    # Apply LLM Extraction
                    llm_labels = extractor.extract_with_llm(failed_texts)
                    
                    # Update DataFrame
                    # Ensure alignment: llm_labels corresponds to failed_indices order
                    df.loc[failed_indices, 'extracted_label'] = llm_labels
            
            # Fill remaining Nones or -1s if any (fallback to incorrect prediction or ignore?)
            # Ensure type is int for metrics
            # If still null (e.g. LLM failed completely?), default to 0
            df['extracted_label'] = df['extracted_label'].fillna(0).astype(int)
            
            # Calculate classification metrics
            # Filter out -1 for final metrics calculation if extractor completely failed?
            # Or treat as wrong? Treating as wrong (e.g. 0 if GT is 1, 1 if GT is 0 -> actually usually just exclude or count as error)
            # Here we will keep them to penalize "I don't know"
            
            y_true = df['ground_truth'].astype(int)
            y_pred = df['extracted_label']
            
            # Metrics
            acc = accuracy_score(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            print(f"  Accuracy: {acc:.4f}, Kappa: {kappa:.4f}, F1: {f1:.4f}")
            
            # Store Global Results
            result_entry = {
                "filename": filename,
                "accuracy": acc,
                "kappa": kappa,
                "f1": f1,
                "n_samples": len(df),
                "n_regex_failed": len(failed_indices)
            }
            global_results.append(result_entry)
            
            # Confusion Matrix Plot
            plot_path = os.path.join(plots_dir, filename.replace('.csv', '_cm.png'))
            # Use 'No DR' and 'DR' labels for 0 and 1
            labels = [0, 1]
            try:
                plot_confusion_matrix(y_true, y_pred, labels=labels, output_path=plot_path, title=f"CM: {filename}")
            except Exception as e:
                print(f"  Could not plot CM: {e}")
            
            # Save detailed report with extractions
            report_path = os.path.join(args.metrics_dir, f"analyzed_{filename}")
            df.to_csv(report_path, index=False)
            print(f"  Saved detailed report to {report_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # traceback.print_exc()

    # Save Global Summary
    if global_results:
        summary_df = pd.DataFrame(global_results)
        summary_path = os.path.join(args.metrics_dir, "summary_metrics.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nAll analysis completed. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
