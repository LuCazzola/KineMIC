import torch
from torch import nn

from typing import Optional, Set
from dataclasses import dataclass

from .moe import LinearMoELoRA

@dataclass
class MoELoRAOptions:
    """Configuration for MoE settings"""
    num_experts: int = 8
    top_k: int = 2
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    

class MoELoRA:
    """Factory class for applying LoRA-MoE to some pytorch model"""
        
    @classmethod
    def _from_linear(cls, module: nn.Linear, opt: MoELoRAOptions) -> LinearMoELoRA:
        """Convert a Linear layer to LoRA-MoE"""
        # Create the LoRA-MoE module matching the given Linear layer.
        moe_module = LinearMoELoRA(
            in_features=module.in_features,
            out_features=module.out_features,
            num_experts=opt.num_experts,
            top_k=opt.top_k,
            r=opt.lora_rank,
            lora_alpha=opt.lora_alpha,
            lora_dropout=opt.lora_dropout,
            bias=module.bias is not None
        )
        # Copy original weights
        moe_module.load_base_weights(module)
        
        return moe_module

    @classmethod
    def from_module(
        cls, 
        module: nn.Module, 
        opt: MoELoRAOptions,
        target_modules: Optional[Set[str]] = None,
        current_name: str = ""
    ) -> nn.Module:
        """Recursively convert target layers to LoRA-MoE"""
        
        # Check if this layer should be converted
        if isinstance(module, nn.Linear):
            # Extract the layer name (last part of the path)
            layer_name = current_name.split('.')[-1] if current_name else ""
            if target_modules is None or any(target in layer_name for target in target_modules):
                print(f"[MoELoRA] Converting {current_name}")
                return cls._from_linear(module, opt)
        
        # Recursively process children
        for name, child in module.named_children():
            child_name = f"{current_name}.{name}" if current_name else name
            module._modules[name] = cls.from_module(
                child, opt, target_modules, child_name
            )
        
        return module
