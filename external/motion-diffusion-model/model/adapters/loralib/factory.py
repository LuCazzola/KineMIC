from dataclasses import dataclass
from typing import Optional, Set
import torch
import torch.nn as nn
from .layers import Linear as LoRALinear, MultiheadAttention as LoRAMultiheadAttention

@dataclass
class LoRAOptions:
    """Configuration for LoRA settings"""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    bias: bool = False
    merge_weights: bool = True
    fan_in_fan_out: bool = False

class LoRA:
    """Factory class for applying LoRA to linear layers"""
    
    def __init__(self):
        pass
    
    @classmethod
    def _from_linear(cls, module: nn.Linear, lora_opt: LoRAOptions) -> LoRALinear:
        """Convert a Linear layer to LoRA"""
        lora_module = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            r=lora_opt.rank,
            lora_alpha=lora_opt.alpha,
            lora_dropout=lora_opt.dropout,
            merge_weights=lora_opt.merge_weights,
            fan_in_fan_out=lora_opt.fan_in_fan_out,
            bias=module.bias is not None
        )
        # Copy original weights
        lora_module.load_base_weights(module, bias_grad=lora_opt.bias)
        
        return lora_module
    
    @classmethod
    def _from_multihead_attention(cls, module: nn.MultiheadAttention, lora_opt: LoRAOptions) -> LoRAMultiheadAttention:
        """Convert a MultiheadAttention layer to LoRA"""
        
        # Create the LoRA attention module.
        lora_module = LoRAMultiheadAttention(
            embed_dim=module.embed_dim,
            num_heads=module.num_heads,
            dropout=module.dropout,  # Use original dropout, not LoRA dropout
            bias=module.in_proj_bias is not None,
            add_bias_kv=module.bias_k is not None,
            add_zero_attn=module.add_zero_attn,
            kdim=module.kdim,
            vdim=module.vdim,
            batch_first=module.batch_first,
            device=next(module.parameters()).device,
            dtype=next(module.parameters()).dtype,
            r=lora_opt.rank,
            lora_alpha=lora_opt.alpha,
            lora_dropout=lora_opt.dropout,  # FIXME: actually unused in current implementation
            merge_weights=lora_opt.merge_weights
        )
        # Copy base weights from given module
        lora_module.load_base_weights(module, bias_grad=lora_opt.bias)
        
        return lora_module

    @classmethod
    def from_module(
        cls, 
        module: nn.Module, 
        lora_opt: LoRAOptions,
        target_modules: Optional[Set[str]] = None,
        current_name: str = ""
    ) -> nn.Module:
        """Recursively convert target layers to LoRA"""
        
        # Check if this layer should be converted
        should_convert = target_modules is None or any(target in current_name for target in target_modules)
        
        if isinstance(module, nn.Linear) and should_convert:
            print(f"[LoRA] Converting {current_name}")
            return cls._from_linear(module, lora_opt)
        
        elif isinstance(module, nn.MultiheadAttention) and should_convert:
            print(f"[LoRA] Converting {current_name}")
            return cls._from_multihead_attention(module, lora_opt)

        # Recursively process children
        for name, child in module.named_children():
            child_name = f"{current_name}.{name}" if current_name else name
            module._modules[name] = cls.from_module(
                child, lora_opt, target_modules, child_name
            )
        
        return module