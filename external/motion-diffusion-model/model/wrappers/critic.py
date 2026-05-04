from torch import nn
from .classifier import Classifier

class Critic(nn.Module):
    """
    A WGAN-GP Critic that outputs a realness score.
    The loss functions have the same interface as a standard GAN Discriminator.
    """
    def __init__(self, model: nn.Module, out_dim: int, **kwargs):
        super().__init__()
        # The Classifier wrapper adds the final linear layer to output a single score
        self.model = Classifier(
            model=model,
            num_classes=1,  # Outputs a single unbounded score
            in_dim=out_dim
        )

    def get_d_loss(self, x_real, x_fake, **kwargs):
        """Calculates the Critic's loss, combining Wasserstein loss and Gradient Penalty."""
        y_real = kwargs.get('y_real', None)
        y_fake = kwargs.get('y_fake', None)
        _, real_scores = self.model(x_real.detach(), y_real)
        _, fake_scores = self.model(x_fake.detach(), y_fake)
        # Discriminator wants to MAXIMIZE the score for real samples and MINIMIZE for fake samples,
        return fake_scores.mean() - real_scores.mean()
        
    def get_g_loss(self, x_fake, **kwargs):
        """Calculates the generator's loss for fooling the critic."""
        _, fake_scores = self.model(x_fake, kwargs.get('y_fake', None))
        # Generator wants to MAXIMIZE the score for its fake samples, so we minimize its negative.
        return -fake_scores.mean()