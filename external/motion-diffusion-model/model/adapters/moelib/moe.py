import torch
import torch.nn as nn
import torch.nn.functional as F

from ..loralib.layers import LoRALayer, LinearDelta


class LinearMoELoRA(nn.Linear, LoRALayer):
    """MoE-LoRA Linear Layer"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 8,
        top_k: int = 2,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        bias: bool = True,
        **kwargs
    ):
        # Initialize base classes
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
        
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        # Freeze base weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        
        # Router
        self.router = nn.Linear(in_features, num_experts, bias=False)
        nn.init.kaiming_uniform_(self.router.weight, a=1.0)
        self.router.weight.data *= 0.1
        
        # LoRA experts using LinearDelta
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = LinearDelta(
                in_features=in_features,
                out_features=out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=0,  # No dropout inside expert (handled externally
            )
            self.experts.append(expert)
    
        # For load balancing
        self._last_routing_weights = None
    
    def reset_parameters(self):
        """Reset parameters"""
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'experts'):
            for expert in self.experts:
                expert.reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with expert routing"""
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        
        # Base transformation
        base_output = F.linear(x_flat, self.weight, self.bias)
                
        # Router
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (torch.sum(top_k_weights, dim=-1, keepdim=True) + 1e-6)
        
        # Expert computation
        expert_output = torch.zeros_like(base_output)
        
        for idx in range(self.num_experts):
            expert_mask = (top_k_indices == idx)
            if not expert_mask.any():
                continue
            
            token_indices, position_indices = torch.where(expert_mask)
            expert_weights = top_k_weights[token_indices, position_indices]
            expert_inputs = x_flat[token_indices]
            
            # Use LinearDelta expert
            expert_inputs = self.lora_dropout(expert_inputs)
            expert_delta = self.experts[idx](expert_inputs)
            expert_output[token_indices] += expert_delta * expert_weights.unsqueeze(-1) # expert contribution
        
        # Store routing weights for load balancing
        if self.training:
            self._last_routing_weights = routing_weights.detach()
        
        # Combine outputs
        final_output = base_output + expert_output
        return final_output.view(original_shape[:-1] + (final_output.shape[-1],))
    
    def get_load_balancing_loss(self) -> torch.Tensor:
        """Compute load balancing loss"""
        if self._last_routing_weights is None or not self.training:
            return torch.tensor(0.0, device=self.router.weight.device, requires_grad=True)
        
        # Routing probability balance
        routing_probs = self._last_routing_weights.mean(dim=0)
        uniform_target = 1.0 / self.num_experts
        prob_loss = ((routing_probs - uniform_target) ** 2).sum()
        
        return prob_loss
    
    def load_base_weights(self, module: nn.Linear):
        with torch.no_grad():
            self.weight.copy_(module.weight)
            if module.bias is not None and self.bias is not None:
                self.bias.copy_(module.bias)