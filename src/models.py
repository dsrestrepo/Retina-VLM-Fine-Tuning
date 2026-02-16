"""
Unified Model Loading and Inference for VLM Evaluation.
"""

import os
import io
import math
import base64
import time
import torch
import warnings
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Tuple
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    MllamaForConditionalGeneration,
    image_utils
)

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

from src.quantization import get_quantization_config

# ============================================================================
# BASE CLASS
# ============================================================================

class VLM(ABC):
    def __init__(
        self, 
        model_id: str, 
        quantization: Optional[str] = None, 
        device: str = "cuda",
        use_flash_attention: bool = True,
        token: Optional[str] = None,
        offline_mode: bool = True,
        adapter_path: Optional[str] = None,
        **kwargs
    ):
        self.model_id = model_id
        self.quantization = quantization
        self.device = device
        self.use_flash_attention = use_flash_attention
        self.token = token or os.getenv("HF_TOKEN")
        self.offline_mode = offline_mode
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        
        # Set offline mode environment variables
        if self.offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        self.load_model(**kwargs)
        self.load_adapter()

    def load_adapter(self):
        """Load LoRA adapter if provided"""
        if self.adapter_path and self.model:
            print(f"Loading adapter from {self.adapter_path}...")
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            except ImportError:
                print("Error: peft library not installed. Cannot load adapter.")
            except Exception as e:
                print(f"Error loading adapter: {e}")

    @abstractmethod
    def load_model(self, **kwargs):
        """Load the model and processor."""
        pass

    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        image: Optional[Union[Image.Image, List[Image.Image]]] = None, 
        return_attention: bool = False, 
        return_logits: bool = False,
        tokens: Optional[List[str]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.01,
        do_sample: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response from the model.
        """
        pass
    
    def _finalize_model(self):
        """Helper to finalize model setup (eval mode, etc)."""
        if self.model:
            # 4b/8b are already on device via device_map="auto"
            if self.quantization not in ["4b", "8b"]:
                self.model.to(self.device)
            self.model.eval()
            torch.cuda.empty_cache()

    def _get_gen_config(self, return_attention, return_logits):
        return {
            "output_attentions": return_attention,
            "output_scores": return_logits,
            "return_dict_in_generate": True,
        }
    
    def format_prompt(self, prompt, image=None, include_image=True):
        """
        Build conversation structure for vision-language models.
        Default implementation for chat template based models.
        Handles both single image and list of images.
        """
        if image is None or not include_image:
            return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            content = []
            images = image if isinstance(image, list) else [image]
            
            # Add image placeholders
            for _ in images:
                content.append({"type": "image"})
            
            # Add text prompt
            content.append({"type": "text", "text": prompt})
            
            return [{"role": "user", "content": content}]

    def extract_token_probs(self, probs, tokens: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract probabilities for specified tokens from probabilities distribution.
        
        Args:
            probs: Probability distribution (tensor) for the first token.
            tokens: List of tokens to extract probabilities for. 
                   If None, defaults to ['yes', 'no', 'Yes', 'No'].
        
        Returns:
            Dict mapping token strings to their probabilities.
        """
        if probs is None:
            return {}
        
        if tokens is None:
            tokens = ['yes', 'no', 'Yes', 'No']
            
        prob_dict = {}
        
        # Ensure we are working with probabilities
        if isinstance(probs, torch.Tensor):
            # If input is logits (often the case with 'scores' output from generate), apply softmax
            # However, if it's already probabilities, this might be wrong. 
            # transformers .generate output_scores are logits.
            probs_tensor = torch.nn.functional.softmax(probs, dim=-1)
        else:
            # Assume it's already a suitable object or list
            return {}

        for token_str in tokens:
            # Tokenize variants (standard, with space, with newline)
            candidates = [token_str, " " + token_str, "\n" + token_str]
            token_prob = 0.0
            found = False
            
            for cand in candidates:
                # We need the tokenizer from the processor
                if hasattr(self.processor, 'tokenizer'):
                    tokenizer = self.processor.tokenizer
                elif hasattr(self.processor, 'encode'): # Some processors have direct encode
                    tokenizer = self.processor
                else:
                    continue

                try:
                    cand_ids = tokenizer.encode(cand, add_special_tokens=False)
                    # We only check single-token candidates for simplicity and accuracy on first token
                    if len(cand_ids) == 1:
                        tid = cand_ids[0]
                        if 0 <= tid < len(probs_tensor):
                            # Get probability
                            val = float(probs_tensor[tid])
                            # We take the max probability among variants (e.g. " yes" vs "yes")
                            token_prob = max(token_prob, val)
                            found = True
                except Exception:
                    continue
            
            prob_dict[token_str] = token_prob if found else None
            
        return prob_dict

    def generate_batch(
        self,
        prompts: List[str],
        images: List[Union[Image.Image, List[Image.Image]]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.01,
        do_sample: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of inputs.
        Default implementation acts as a loop wrapper, but subclasses should override for true batching.
        """
        results = []
        # Normalization
        if images is None:
            images = [None] * len(prompts)
        elif len(images) != len(prompts):
            # If explicit images provided but length mismatch, handle?
            # Assuming aligned if list provided
            pass
            
        for p, i in zip(prompts, images):
            res = self.generate(
                prompt=p,
                image=i,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs
            )
            results.append(res["text"])
        return results

    def clean_memory(self):
        """Clean GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# ============================================================================
# QWEN MODEL
# ============================================================================

class QwenVLM(VLM):
    def load_model(self, **kwargs):
        bnb_config, torch_dtype = get_quantization_config(self.quantization, use_bfloat16=True)
        
        if self.use_flash_attention and self.quantization is None:
            torch_dtype = torch.bfloat16

        model_kwargs = {
            "device_map": "auto",
            "quantization_config": bnb_config,
            "torch_dtype": torch_dtype,
            "local_files_only": self.offline_mode,
        }
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            
        # Determine model class based on ID
        if "Qwen3" in self.model_id:
             if Qwen3VLForConditionalGeneration is None:
                 raise ImportError("Qwen3VLForConditionalGeneration not found in transformers. Update transformers library.")
             ModelClass = Qwen3VLForConditionalGeneration
        else:
             ModelClass = Qwen2VLForConditionalGeneration

        # Use AutoProcessor for both Qwen2 and Qwen3
        ProcessorClass = AutoProcessor

        self.model = ModelClass.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        
        self.model = self.model.eval() # Manual finalize since we use device_map="auto" usually
        if self.quantization not in ["4b", "8b"]:
            # If not quantized, ensure it's on device if device_map didn't handle it (though usually auto does)
            pass 

        self.processor = ProcessorClass.from_pretrained(self.model_id, local_files_only=self.offline_mode)
        
    def generate(self, prompt, image=None, return_attention=False, return_logits=False, tokens=None, max_new_tokens=128, temperature=0.01, do_sample=False, **kwargs):
        # Qwen specific conversation format
        conversation = self.format_prompt(prompt, image)
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Handle single or multiple images
        images_list = None
        if image is not None:
             images_list = image if isinstance(image, list) else [image]

        inputs = self.processor(
            text=[text_prompt], 
            images=images_list, 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        gen_kwargs = self._get_gen_config(return_attention, return_logits)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                do_sample=do_sample,
                **gen_kwargs
            )
            
        output_ids = outputs.sequences[0]
        input_len = inputs.input_ids.shape[-1]
        generated_ids = output_ids[input_len:]
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # Extract token probs
        token_probs = {}
        if return_logits and tokens:
            probs = outputs.scores[0][0]
            token_probs = self.extract_token_probs(probs, tokens)
        
        return {
            "text": output_text,
            "generated_ids": [generated_ids],
            "attentions": outputs.attentions if return_attention else None,
            "scores": outputs.scores if return_logits else None,
            "token_probs": token_probs
        }

    def generate_batch(self, prompts: List[str], images: List[Union[Image.Image, List[Image.Image]]] = None, max_new_tokens: int = 128, temperature: float = 0.01, do_sample: bool = False, **kwargs):
        processed_prompts = []
        if images is None:
            images = [None] * len(prompts)
        
        for p, img in zip(prompts, images):
            conversation = self.format_prompt(p, img)
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            processed_prompts.append(text_prompt)
        
        # Structure: [[img1, img2], [img3], ...]
        nested_images = []
        for img_item in images:
            if img_item is not None:
                if isinstance(img_item, list):
                    nested_images.append(img_item)
                else:
                    nested_images.append([img_item])
            else:
                nested_images.append([])

        inputs = self.processor(
            text=processed_prompts, 
            images=nested_images if any(nested_images) else None, 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        gen_kwargs = self._get_gen_config(False, False)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                do_sample=do_sample,
                **gen_kwargs
            )
            
        output_ids = outputs.sequences
        input_len = inputs.input_ids.shape[-1]
        generated_ids = output_ids[:, input_len:]
        output_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return output_texts

# ============================================================================
# LLAVA MODEL
# ============================================================================

class LlavaVLM(VLM):
    def load_model(self, **kwargs):
        bnb_config, torch_dtype = get_quantization_config(self.quantization, use_bfloat16=True)
        
        if self.use_flash_attention and self.quantization is None:
            torch_dtype = torch.bfloat16

        is_v15 = "1.5" in self.model_id
        ModelClass = LlavaForConditionalGeneration if is_v15 else LlavaNextForConditionalGeneration
        ProcessorClass = AutoProcessor if is_v15 else LlavaNextProcessor
        
        model_kwargs = {
            "device_map": "auto",
            "quantization_config": bnb_config,
            "torch_dtype": torch_dtype,
            "local_files_only": self.offline_mode,
        }
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = ModelClass.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        self._finalize_model()
        self.processor = ProcessorClass.from_pretrained(self.model_id, local_files_only=self.offline_mode)

    def generate(self, prompt, image=None, return_attention=False, return_logits=False, tokens=None, max_new_tokens=128, temperature=0.01, do_sample=False, **kwargs):
        conversation = self.format_prompt(prompt, image)
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Handle single or multiple images
        images_list = None
        if image is not None:
             images_list = image if isinstance(image, list) else [image]
        
        inputs = self.processor(
            text=[text_prompt], 
            images=images_list, 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        gen_kwargs = self._get_gen_config(return_attention, return_logits)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                do_sample=do_sample,
                **gen_kwargs
            )
            
        output_ids = outputs.sequences[0]
        input_len = inputs.input_ids.shape[-1]
        generated_ids = output_ids[input_len:]
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        token_probs = {}
        if return_logits and tokens:
            probs = outputs.scores[0][0]
            token_probs = self.extract_token_probs(probs, tokens)
        
        return {
            "text": output_text,
            "generated_ids": [generated_ids],
            "attentions": outputs.attentions if return_attention else None,
            "scores": outputs.scores if return_logits else None,
            "token_probs": token_probs
        }

    def generate_batch(self, prompts: List[str], images: List[Union[Image.Image, List[Image.Image]]] = None, max_new_tokens: int = 128, temperature: float = 0.01, do_sample: bool = False, **kwargs):
        processed_prompts = []
        if images is None:
            images = [None] * len(prompts)
            
        for p, img in zip(prompts, images):
            conversation = self.format_prompt(p, img)
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            processed_prompts.append(text_prompt)
            
        # Use nested list for images
        nested_images = []
        for img_item in images:
            if img_item is not None:
                if isinstance(img_item, list):
                    nested_images.append(img_item)
                else:
                    nested_images.append([img_item])
            else:
                nested_images.append([])

        if not any(nested_images):
            inputs = self.processor(text=processed_prompts, padding=True, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor(text=processed_prompts, images=nested_images, padding=True, return_tensors="pt").to(self.model.device)

        gen_kwargs = self._get_gen_config(False, False)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                do_sample=do_sample,
                **gen_kwargs
            )
            
        output_ids = outputs.sequences
        input_len = inputs.input_ids.shape[-1]
        generated_ids = output_ids[:, input_len:]
        output_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return output_texts

# ============================================================================
# LLAMA MODEL
# ============================================================================

class LlamaVLM(VLM):
    def load_model(self, **kwargs):
        bnb_config, torch_dtype = get_quantization_config(self.quantization, use_bfloat16=True)
        
        if self.use_flash_attention and self.quantization is None:
            torch_dtype = torch.bfloat16

        # 16b check logic from reference
        if self.quantization == "16b":
             torch_dtype = torch.float16 # Llama prefers float16? Reference said so.

        model_kwargs = {
            "device_map": "auto",
            "token": self.token,
            "quantization_config": bnb_config,
            "torch_dtype": torch_dtype,
            "local_files_only": self.offline_mode,
        }
        if self.use_flash_attention:
            print(f"[{self.model_id}] Using 'sdpa' attention implementation (Flash Attention 2 fallback).")
            model_kwargs["attn_implementation"] = "sdpa"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        self._finalize_model()
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=self.token, local_files_only=self.offline_mode)

    def generate(self, prompt, image=None, return_attention=False, return_logits=False, tokens=None, unmatched=False, max_new_tokens=128, temperature=0.01, do_sample=False, **kwargs):
        # MLLaMA specific suffix logic
        suffix = " Only respond with the answer (yes, no, or unmatched). No aditional commentary." if unmatched else " Only respond with the answer (yes or no). No aditional commentary."
        prompt = prompt + suffix
        
        conversation = self.format_prompt(prompt, image)
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Handle single or multiple images
        images_list = None
        if image is not None:
             images_list = image if isinstance(image, list) else [image]

        inputs = self.processor(
            text=[text_prompt], 
            images=images_list, 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        gen_kwargs = self._get_gen_config(return_attention, return_logits)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                do_sample=do_sample,
                **gen_kwargs
            )
            
        output_ids = outputs.sequences[0]
        input_len = inputs.input_ids.shape[-1]
        generated_ids = output_ids[input_len:]
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        token_probs = {}
        if return_logits and tokens:
            probs = outputs.scores[0][0]
            token_probs = self.extract_token_probs(probs, tokens)
        
        return {
            "text": output_text,
            "generated_ids": [generated_ids],
            "attentions": outputs.attentions if return_attention else None,
            "scores": outputs.scores if return_logits else None,
            "token_probs": token_probs
        }

    def generate_batch(self, prompts: List[str], images: List[Union[Image.Image, List[Image.Image]]] = None, max_new_tokens: int = 128, temperature: float = 0.01, do_sample: bool = False, **kwargs):
        processed_prompts = []
        if images is None:
            images = [None] * len(prompts)
            
        for p, img in zip(prompts, images):
            conversation = self.format_prompt(p, img)
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            processed_prompts.append(text_prompt)
            
        # For MLLaMA, images must be a nested list if we have a batch of prompts
        # Structure: [[img1_p1], [img1_p2], ...]
        nested_images = []
        for img_item in images:
            if img_item is not None:
                if isinstance(img_item, list):
                    nested_images.append(img_item)
                else:
                    nested_images.append([img_item])
            else:
                nested_images.append([])

        inputs = self.processor(
            text=processed_prompts, 
            images=nested_images if any(nested_images) else None, 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        gen_kwargs = self._get_gen_config(False, False)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                do_sample=do_sample,
                **gen_kwargs
            )
            
        output_ids = outputs.sequences
        input_len = inputs.input_ids.shape[-1]
        generated_ids = output_ids[:, input_len:]
        output_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return output_texts

# ============================================================================
# GEMMA / MEDGEMMA MODEL
# ============================================================================

class GemmaVLM(VLM):
    def load_model(self, **kwargs):
        bnb_config, torch_dtype = get_quantization_config(self.quantization, use_bfloat16=True)
        
        if self.use_flash_attention and self.quantization is None:
            torch_dtype = torch.bfloat16

        model_kwargs = {
            "device_map": kwargs.get("device_map", "auto"),
            "quantization_config": bnb_config,
            "torch_dtype": torch_dtype,
            "token": self.token,
            "local_files_only": self.offline_mode,
        }
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        self._finalize_model()
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=self.token, local_files_only=self.offline_mode)

    def format_prompt(self, prompt, image, include_image=True, **kwargs):
        # Specific override for Gemma structure
        if image is None:
            return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            content = []
            images = image if isinstance(image, list) else [image]
            
            # Use slice markers if requested, or default to True if multiple images
            # This matches the logic from inference.py
            include_slice_markers = kwargs.get("include_slice_markers")
            use_markers = include_slice_markers if include_slice_markers is not None else (len(images) > 1)
            
            for i, img in enumerate(images):
                content.append({"type": "image", "image": img})
                if use_markers:
                    content.append({"type": "text", "text": f"SLICE {i+1}"})
            
            content.append({"type": "text", "text": prompt})
                
            return [{"role": "user", "content": content}]

    def generate(self, prompt, image=None, return_attention=False, return_logits=False, tokens=None, max_new_tokens=512, temperature=0.01, do_sample=False, **kwargs):
        # Gemma 3 / MedGemma specific structure from inference.py
        messages = self.format_prompt(prompt, image, **kwargs)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Casting to appropriate dtype if quantized
        dtype = torch.bfloat16 if self.quantization in ["4b", "8b"] else None
        inputs = inputs.to(self.model.device, dtype=dtype)
        
        gen_kwargs = self._get_gen_config(return_attention, return_logits)
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **gen_kwargs
            )
            
        output_ids = outputs.sequences[0][input_len:]
        output_text = self.processor.decode(output_ids, skip_special_tokens=True).strip()
        
        token_probs = {}
        if return_logits and tokens:
            probs = outputs.scores[0][0]
            token_probs = self.extract_token_probs(probs, tokens)
        
        return {
            "text": output_text,
            "generated_ids": [output_ids],
            "attentions": outputs.attentions if return_attention else None,
            "scores": outputs.scores if return_logits else None,
            "token_probs": token_probs
        }

    def generate_batch(self, prompts: List[str], images: List[Union[Image.Image, List[Image.Image]]] = None, max_new_tokens: int = 128, temperature: float = 0.01, do_sample: bool = False, **kwargs):
        if images is None:
            images = [None] * len(prompts)
            
        batch_conversations = []
        for p, img in zip(prompts, images):
            messages = self.format_prompt(p, img, **kwargs)
            batch_conversations.append(messages)

        inputs = self.processor.apply_chat_template(
            batch_conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        dtype = torch.bfloat16 if self.quantization in ["4b", "8b"] else None
        inputs = inputs.to(self.model.device, dtype=dtype)
        
        gen_kwargs = self._get_gen_config(False, False)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **gen_kwargs
            )
            
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs.sequences[:, input_len:]
        output_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return [text.strip() for text in output_texts]

# ============================================================================
# OPENAI MODEL
# ============================================================================

class OpenAIVLM(VLM):
    def load_model(self, **kwargs):
        from openai import OpenAI
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model_name = self.model_id # e.g. "gpt-4o"

    def _pil_to_base64_jpeg(self, img, quality=85):
        """Return a data URL 'data:image/jpeg;base64,...' from a PIL.Image."""
        if img is None:
            return None
        if not isinstance(img, Image.Image):
            img = image_utils.load_image(img)
        # Convert to RGB to ensure JPEG compatibility (no alpha channel)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def format_prompt(self, prompt, image=None, include_image=True):
        is_gpt5 = "gpt-5" in self.model_name
        
        if image is None or not include_image:
            return [{"role": "user", "content": prompt}]
        
        content = []
        images = image if isinstance(image, list) else [image]
        
        if is_gpt5:
             content.append({"type": "input_text", "text": prompt})
             for img in images:
                 data_url = self._pil_to_base64_jpeg(img)
                 content.append({"type": "input_image", "image_url": data_url})
        else:
             content.append({"type": "text", "text": prompt})
             for img in images:
                 data_url = self._pil_to_base64_jpeg(img)
                 content.append({"type": "image_url", "image_url": {"url": data_url}})
        return [{"role": "user", "content": content}]

    def generate(self, prompt, image=None, return_attention=False, return_logits=False, tokens=None, max_new_tokens=128, temperature=0.01, do_sample=False, max_retries=5, **kwargs):
        # Limited support for attention/logits in API generally
        messages = self.format_prompt(prompt, image)
        
        is_reasoning = "o1" in self.model_name or "o3" in self.model_name or "reasoning" in self.model_name
        is_gpt5 = "gpt-5" in self.model_name

        # Prepare parameters
        api_kwargs = {}
        if is_reasoning or is_gpt5:
            # Reasoning models use max_completion_tokens and strict temperature policies
            api_kwargs["max_completion_tokens"] = max_new_tokens
        else:
            api_kwargs["max_tokens"] = max_new_tokens
            api_kwargs["temperature"] = temperature
        
        api_kwargs["timeout"] = 90.0

        # Retry logic with exponential backoff
        retry_count = 0
        completion = None
        out_text = ""
        
        while retry_count < max_retries:
            try:
                if is_gpt5:
                    resp = self.client.responses.create(
                        model=self.model_name,
                        input=messages,
                        reasoning={"effort": "low"},
                        timeout=90.0
                    )
                    out_text = resp.output_text
                    completion = resp
                else:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **api_kwargs
                    )
                    out_text = completion.choices[0].message.content.strip()
                
                break
            except Exception as e:
                retry_count += 1
                error_str = str(e)
                
                # Check for rate limits or other specific errors
                if "429" in error_str:
                    # Rate limit: exponential backoff
                    wait_time = min(60, 2 * (2 ** (retry_count - 1)))
                    print(f"OpenAI Rate Limit Hit. Waiting {wait_time}s...")
                elif "timeout" in error_str.lower():
                    # Timeouts might be transient
                    wait_time = 5
                    print(f"OpenAI Timeout (Attempt {retry_count}/{max_retries}). Retrying in {wait_time}s...")
                else:
                    # Other errors
                    wait_time = min(60, 1 * (2 ** (retry_count - 1)))
                    print(f"Error in OpenAI API (Attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Skipping...")
                    return {"text": "", "error": str(e)}

        return {
            "text": out_text
        }

# ============================================================================
# GEMINI MODEL
# ============================================================================

class GeminiVLM(VLM):
    def load_model(self, **kwargs):
        from google import genai
        
        self.vertex_api = kwargs.get("vertex_api", True)
        
        if self.vertex_api:
            key = kwargs.get("project_id") or os.getenv("GEMINI_API_PROJECT_ID")
            if (not key) or (not os.getenv("GOOGLE_APPLICATION_CREDENTIALS")):
                # Fallback or warning? References raises ValueError.
                # Assuming environment is set up as expected.
                pass
            self.client = genai.Client(vertexai=True, project=key, location="global")
        else:
            key = kwargs.get("api_key") or os.getenv("GEMINI_API_KEY")
            self.client = genai.Client(api_key=key)
    
    def _pil_to_base64_png(self, img):
        import io
        if not isinstance(img, Image.Image):
            img = image_utils.load_image(img)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return b64

    def format_prompt(self, prompt, image=None, include_image=True):
        if image is None or not include_image:
            return [prompt]
        
        parts = [prompt]
        images = image if isinstance(image, list) else [image]
        
        for img in images:
            b64_data = self._pil_to_base64_png(img)
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64_data}})
            
        return parts

    def generate(self, prompt, image=None, return_attention=False, return_logits=False, tokens=None, max_new_tokens=128, temperature=0.01, do_sample=False, max_retries=5, **kwargs):
        from google.genai import types
        import logging
        
        # Suppress logging warnings from google.genai as per reference behavior
        logging.getLogger("google.genai").setLevel(logging.ERROR)

        parts = self.format_prompt(prompt, image)
        
        # Logic for thinking config based on model ID (Gemini 3 specific)
        is_gemini_3 = "gemini-3" in self.model_id
        thinking_budget = 1 if is_gemini_3 else 0
        if is_gemini_3 and max_new_tokens < 256:
            print('Warning: Gemini 3 models may require higher max_new_tokens (>=256) for proper reasoning.')
        
        # Config
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=thinking_budget)
        )

        # Retry logic: wait 60 seconds on error
        retry_count = 0
        response = None
        
        while retry_count < max_retries:
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=parts,
                    config=gen_config
                )
                break
            except Exception as e:
                retry_count += 1
                print(f"Error in Gemini API (Attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print("Waiting 60 seconds before retrying...")
                    time.sleep(60)
                else:
                    print("Max retries reached. Skipping...")
                    return {"text": "", "error": str(e)}

        # Response extraction
        out_text = ""
        try:
             if response:
                 if response.text:
                     out_text = response.text
                 elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                     # Check if candidates exist but .text is empty/None
                     parts = response.candidates[0].content.parts
                     out_text = "".join([part.text for part in parts if hasattr(part, 'text') and part.text])
        except ValueError:
            pass
        except Exception as e:
            print(f"Gemini Extraction Error: {e}")

        return {
            "text": out_text
        }

# ============================================================================
# GPT OSS MODEL (Text Only Support)
# ============================================================================

class GptOssLLM(VLM):
    def load_model(self, **kwargs):
        # GPT-OSS models typically use native MXFP4 quantization or loaded in BFloat16 with kernels
        # B&B quantization is generally not compatible or redundant.
        
        # Decide usage of optimized kernels (Liger RMSNorm, MegaBlocks MoE) based on quantization.
        # According to documentation: "These kernels are not compatible with mxfp4"
        # - If user requests quantization ("4b"/"8b"): We disable use_kernels to allow native MXFP4 (low memory).
        # - If user requests standard (None/"16b"): We enable use_kernels for optimization (BFloat16).
        
        use_kernels = False
        if self.quantization in ["4b", "8b"]:
             print(f"[{self.model_id}] Quantization requested. Disabling 'use_kernels' to allow native MXFP4/Efficient loading.")
             use_kernels = False
        else:
             print(f"[{self.model_id}] No quantization requested. Enabling 'use_kernels' for optimization (BFloat16).")
             use_kernels = True

        torch_dtype = torch.bfloat16

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch_dtype,
                local_files_only=self.offline_mode,
                use_kernels=use_kernels,
            )
        except Exception as e:
            if use_kernels:
                print(f"[{self.model_id}] Failed to load with use_kernels=True (Error: {e}). Retrying with use_kernels=False...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                    local_files_only=self.offline_mode,
                    use_kernels=False,
                )
            else:
                raise e
        
        self._finalize_model()
        self.processor = AutoTokenizer.from_pretrained(self.model_id, local_files_only=self.offline_mode)

    def generate(self, prompt, image=None, return_attention=False, return_logits=False, tokens=None, max_new_tokens=128, temperature=0.01, do_sample=False, reasoning_effort="medium", **kwargs):
        if image is not None:
            # GPT OSS provided description is text-generation only
            print(f"Warning: {self.model_id} does not support images. Image ignored.")

        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=reasoning_effort
        ).to(self.model.device)

        gen_kwargs = self._get_gen_config(return_attention, return_logits)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                do_sample=do_sample,
                **gen_kwargs
            )
            
        output_ids = outputs.sequences[0]
        input_len = inputs.input_ids.shape[-1]
        generated_ids = output_ids[input_len:]
        
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        token_probs = {}
        if return_logits and tokens:
            probs = outputs.scores[0][0]
            token_probs = self.extract_token_probs(probs, tokens)
        
        return {
            "text": output_text,
            "generated_ids": [generated_ids],
            "attentions": outputs.attentions if return_attention else None,
            "scores": outputs.scores if return_logits else None,
            "token_probs": token_probs
        }
