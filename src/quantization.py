"""
Quantization utilities for model loading.

This module provides utilities for configuring model quantization using BitsAndBytes.
"""

import torch
from typing import Optional, Tuple

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


def get_quantization_config(
    quantization: Optional[str],
    use_bfloat16: bool = True,
) -> Tuple[Optional[object], torch.dtype]:
    """
    Get quantization configuration and torch dtype for model loading.
    
    Args:
        quantization: Quantization type ("4b", "8b", "16b", or None)
        use_bfloat16: Whether to use bfloat16 for quantized models (default: True)
                     Some models require float16 instead
    
    Returns:
        Tuple of (BitsAndBytesConfig or None, torch_dtype)
    """
    if BitsAndBytesConfig is None or quantization is None:
        return None, torch.float32
    
    compute_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    
    if quantization == "4b":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        return bnb_config, compute_dtype
    
    elif quantization == "8b":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=compute_dtype,
        )
        return bnb_config, compute_dtype
    
    elif quantization == "16b":
        # No BitsAndBytes config for 16b, just use appropriate dtype
        return None, compute_dtype
    
    # No quantization
    return None, torch.float32


def is_quantized(quantization: Optional[str]) -> bool:
    """Check if quantization requires BitsAndBytes (4b or 8b)."""
    return quantization in ["4b", "8b"]


def get_model_dtype(quantization: Optional[str], use_bfloat16: bool = True) -> torch.dtype:
    """
    Get the appropriate torch dtype for a given quantization setting.
    
    Args:
        quantization: Quantization type ("4b", "8b", "16b", or None)
        use_bfloat16: Whether to use bfloat16 for 16b quantization
    
    Returns:
        torch.dtype appropriate for the quantization setting
    """
    if quantization in ["4b", "8b"]:
        return torch.bfloat16 if use_bfloat16 else torch.float16
    elif quantization == "16b":
        return torch.bfloat16 if use_bfloat16 else torch.float16
    else:
        return torch.float32
