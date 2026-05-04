from torch import nn
import torch

def gradient_penalty(model, x_real, x_fake, y):
        """Calculates the gradient penalty for WGAN-GP to enforce the Lipschitz constraint."""
        B = x_real.shape[0]
        # Create a random interpolation factor alpha
        shape_broadcast = (B,) + (1,) * (x_real.dim() - 1)
        alpha = torch.rand(shape_broadcast, device=x_real.device)
        # Interpolate between real and fake samples
        interpolated = (alpha * x_real.detach() + (1 - alpha) * x_fake.detach()).requires_grad_(True)
        # Pass the metadata to the model for the interpolated samples
        _, interpolated_scores = model(interpolated, y)
        # Calculate the gradients of the scores with respect to the interpolated inputs
        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True, retain_graph=False,
        )[0]
        # Reshape gradients and calculate the L2 norm
        gradients = gradients.view(B, -1)
        gradient_norm = gradients.norm(2, dim=1)
        # The penalty is the mean squared difference between the norm and 1
        return ((gradient_norm - 1) ** 2).mean()

def spectral_norm(module: nn.Module):
    """
    Applies Spectral Normalization only to relevant modules (Conv/Linear layers)
    that have a 'weight' parameter, and skips others (like BatchNorm).
    
    This function is designed to be passed to nn.Module.apply().
    
    Args:
        module (nn.Module): The module to check and apply SN to.
    """

    RELEVANT_MODULES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    EXCLUDED_MODULES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    
    if isinstance(module, EXCLUDED_MODULES):
        return
    if not isinstance(module, RELEVANT_MODULES):
        return

    if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
        is_sn_applied = any(name.startswith('weight_orig') for name, _ in module.named_parameters(recurse=False))
        if not is_sn_applied:
            try:
                torch.nn.utils.spectral_norm(module)
            except ValueError:
                print(f"Spectral Norm could not be applied to {module.__class__.__name__}")
