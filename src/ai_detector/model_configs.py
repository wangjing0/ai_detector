"""
Model configurations for AI text detection.

This module contains predefined configurations for different open source models
that can be used as observer and performer models in the AI detector.
"""

from typing import Dict, List, Tuple
import warnings

# Model configurations with compatible tokenizer pairs
MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "llama_3_1_8b": {
        "observer": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "performer": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "description": "Meta Llama 3.1 8B base and instruction-tuned models"
    },
    
    "llama_3_1_70b": {
        "observer": "unsloth/Meta-Llama-3.1-70B-bnb-4bit", 
        "performer": "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "description": "Meta Llama 3.1 70B base and instruction-tuned models (requires more VRAM)"
    },
    
    "llama_3_2_3b": {
        "observer": "unsloth/Llama-3.2-3B-bnb-4bit",
        "performer": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit", 
        "description": "Meta Llama 3.2 3B base and instruction-tuned models (lightweight)"
    },
    
    "llama_2_7b": {
        "observer": "NousResearch/Llama-2-7b-hf",
        "performer": "NousResearch/Llama-2-7b-chat-hf",
        "description": "Meta Llama 2 7B base and chat models (full precision, will auto-quantize)"
    },
    
    "llama_2_13b": {
        "observer": "NousResearch/Llama-2-13b-hf", 
        "performer": "NousResearch/Llama-2-13b-chat-hf",
        "description": "Meta Llama 2 13B base and chat models (full precision, will auto-quantize)"
    },
    
    "mistral_7b": {
        "observer": "mistralai/Mistral-7B-v0.1",
        "performer": "mistralai/Mistral-7B-Instruct-v0.1",
        "description": "Mistral 7B base and instruction-tuned models"
    },
    
    "mistral_7b_v3": {
        "observer": "mistralai/Mistral-7B-v0.3",
        "performer": "mistralai/Mistral-7B-Instruct-v0.3", 
        "description": "Mistral 7B v0.3 base and instruction-tuned models"
    },
    
    "phi_3_mini": {
        "observer": "microsoft/Phi-3-mini-4k-instruct",
        "performer": "microsoft/Phi-3-mini-128k-instruct",
        "description": "Microsoft Phi-3 Mini models with different context lengths"
    },
    
    "phi_3_medium": {
        "observer": "microsoft/Phi-3-medium-4k-instruct", 
        "performer": "microsoft/Phi-3-medium-128k-instruct",
        "description": "Microsoft Phi-3 Medium models (requires more VRAM)"
    },
    
    "gemma_2b": {
        "observer": "google/gemma-2b",
        "performer": "google/gemma-2b-it",
        "description": "Google Gemma 2B base and instruction-tuned models (lightweight)"
    },
    
    "gemma_7b": {
        "observer": "google/gemma-7b", 
        "performer": "google/gemma-7b-it",
        "description": "Google Gemma 7B base and instruction-tuned models"
    },
    
    "qwen2_7b": {
        "observer": "Qwen/Qwen2-7B",
        "performer": "Qwen/Qwen2-7B-Instruct",
        "description": "Qwen2 7B base and instruction-tuned models"
    },
    
    "code_llama_7b": {
        "observer": "codellama/CodeLlama-7b-hf",
        "performer": "codellama/CodeLlama-7b-Instruct-hf", 
        "description": "Meta Code Llama 7B for code-focused detection"
    },
    
    "falcon_7b": {
        "observer": "tiiuae/falcon-7b",
        "performer": "tiiuae/falcon-7b-instruct",
        "description": "TII Falcon 7B base and instruction-tuned models"
    }
}

# Model size categories for easy selection
MODEL_SIZES: Dict[str, List[str]] = {
    "small": ["phi_3_mini", "gemma_2b", "llama_3_2_3b"],
    "medium": ["mistral_7b", "mistral_7b_v3", "llama_2_7b", "gemma_7b", "qwen2_7b", "code_llama_7b", "falcon_7b", "llama_3_1_8b"],
    "large": ["llama_2_13b", "phi_3_medium"],
    "extra_large": ["llama_3_1_70b"]
}

# Performance characteristics (estimated)
MODEL_PERFORMANCE: Dict[str, Dict[str, str]] = {
    "llama_3_1_8b": {"speed": "medium", "accuracy": "high", "vram": "medium"},
    "llama_3_1_70b": {"speed": "slow", "accuracy": "very_high", "vram": "very_high"},
    "llama_3_2_3b": {"speed": "fast", "accuracy": "medium", "vram": "low"},
    "llama_2_7b": {"speed": "medium", "accuracy": "medium", "vram": "medium"},
    "llama_2_13b": {"speed": "slow", "accuracy": "high", "vram": "high"}, 
    "mistral_7b": {"speed": "medium", "accuracy": "high", "vram": "medium"},
    "mistral_7b_v3": {"speed": "medium", "accuracy": "high", "vram": "medium"},
    "phi_3_mini": {"speed": "fast", "accuracy": "medium", "vram": "low"},
    "phi_3_medium": {"speed": "slow", "accuracy": "high", "vram": "high"},
    "gemma_2b": {"speed": "very_fast", "accuracy": "low", "vram": "very_low"},
    "gemma_7b": {"speed": "medium", "accuracy": "medium", "vram": "medium"},
    "qwen2_7b": {"speed": "medium", "accuracy": "high", "vram": "medium"},
    "code_llama_7b": {"speed": "medium", "accuracy": "medium_code", "vram": "medium"},
    "falcon_7b": {"speed": "medium", "accuracy": "medium", "vram": "medium"}
}


def get_model_config(config_name: str) -> Tuple[str, str]:
    """
    Get observer and performer model paths for a given configuration.
    
    Args:
        config_name: Name of the model configuration
        
    Returns:
        Tuple of (observer_model_path, performer_model_path)
        
    Raises:
        ValueError: If config_name is not found
    """
    if config_name not in MODEL_CONFIGS:
        available_configs = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Model config '{config_name}' not found. Available configs: {available_configs}")
    
    config = MODEL_CONFIGS[config_name]
    return config["observer"], config["performer"]


def list_available_configs() -> List[str]:
    """Get list of available model configuration names."""
    return list(MODEL_CONFIGS.keys())


def get_configs_by_size(size: str) -> List[str]:
    """
    Get model configurations by size category.
    
    Args:
        size: Size category ('small', 'medium', 'large', 'extra_large')
        
    Returns:
        List of configuration names for that size category
    """
    if size not in MODEL_SIZES:
        raise ValueError(f"Size '{size}' not found. Available sizes: {list(MODEL_SIZES.keys())}")
    return MODEL_SIZES[size]


def get_model_info(config_name: str) -> Dict[str, str]:
    """
    Get detailed information about a model configuration.
    
    Args:
        config_name: Name of the model configuration
        
    Returns:
        Dictionary with model information including paths, description, and performance
    """
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Model config '{config_name}' not found.")
    
    config = MODEL_CONFIGS[config_name].copy()
    if config_name in MODEL_PERFORMANCE:
        config.update(MODEL_PERFORMANCE[config_name])
    
    return config


def recommend_config(vram_gb: float = None, speed_priority: bool = False) -> str:
    """
    Recommend a model configuration based on system constraints.
    
    Args:
        vram_gb: Available VRAM in GB (optional)
        speed_priority: Whether to prioritize inference speed
        
    Returns:
        Recommended configuration name
    """
    if speed_priority:
        if vram_gb and vram_gb < 8:
            return "gemma_2b"  # Fastest, lowest VRAM
        else:
            return "llama_3_2_3b"  # Good balance of speed and accuracy
    
    if vram_gb:
        if vram_gb < 6:
            return "gemma_2b"
        elif vram_gb < 12:
            return "llama_3_2_3b" 
        elif vram_gb < 24:
            return "llama_3_1_8b"
        else:
            return "llama_3_1_70b"
    
    # Default recommendation
    return "llama_3_1_8b"


def validate_config_compatibility(observer_model: str, performer_model: str) -> bool:
    """
    Validate that two models have compatible tokenizers.
    
    Args:
        observer_model: Path to observer model
        performer_model: Path to performer model
        
    Returns:
        True if models are likely compatible, False otherwise
        
    Note:
        This is a heuristic check based on model family names.
        The actual tokenizer compatibility check happens in the Detector class.
    """
    # Extract model family from paths
    def get_model_family(model_path: str) -> str:
        path_lower = model_path.lower()
        if "llama" in path_lower:
            if "3.1" in path_lower or "3.2" in path_lower:
                return "llama3"
            elif "2" in path_lower:
                return "llama2" 
            else:
                return "llama"
        elif "mistral" in path_lower:
            return "mistral"
        elif "phi" in path_lower:
            return "phi"
        elif "gemma" in path_lower:
            return "gemma"
        elif "qwen" in path_lower:
            return "qwen"
        elif "falcon" in path_lower:
            return "falcon"
        elif "code" in path_lower:
            return "codellama"
        else:
            return "unknown"
    
    observer_family = get_model_family(observer_model)
    performer_family = get_model_family(performer_model)
    
    # Same family should be compatible
    if observer_family == performer_family:
        return True
    
    # Some cross-family compatibilities (heuristic)
    compatible_pairs = {
        ("llama2", "llama3"),
        ("llama3", "llama2"), 
        ("llama", "codellama"),
        ("codellama", "llama")
    }
    
    return (observer_family, performer_family) in compatible_pairs


if __name__ == "__main__":
    # Example usage
    print("Available model configurations:")
    for config_name in list_available_configs():
        info = get_model_info(config_name)
        print(f"  {config_name}: {info['description']}")
        if config_name in MODEL_PERFORMANCE:
            perf = MODEL_PERFORMANCE[config_name]
            print(f"    Speed: {perf['speed']}, Accuracy: {perf['accuracy']}, VRAM: {perf['vram']}")
    
    print(f"\nRecommended config for 8GB VRAM: {recommend_config(8.0)}")
    print(f"Recommended config for speed: {recommend_config(speed_priority=True)}")
    
    # Test a configuration
    observer, performer = get_model_config("llama_3_1_8b")
    print(f"\nUsing llama_3_1_8b config:")
    print(f"  Observer: {observer}")
    print(f"  Performer: {performer}")