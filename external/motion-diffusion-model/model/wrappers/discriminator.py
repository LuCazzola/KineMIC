import torch
from torch import nn
from .classifier import Classifier

class Discriminator(nn.Module):

    def __init__(self, model : nn.Module, out_dim : int, **kwargs):
        super().__init__()
        self.model = Classifier(
            model=model,
            num_classes=1,
            in_dim=out_dim
        )
        # Binary classification (real/fake)
        self.criterion = nn.BCEWithLogitsLoss()

    def get_d_loss(self, x_real, x_fake, **kwargs):
        """Calculates the discriminator loss."""
        _, real_logits = self.model(x_real.detach(), kwargs.get('y_real', None))
        _, fake_logits = self.model(x_fake.detach(), kwargs.get('y_fake', None))
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        real_loss = self.criterion(real_logits, real_labels)
        fake_loss = self.criterion(fake_logits, fake_labels)
        return (real_loss + fake_loss) / 2

    def get_g_loss(self, x_fake, **kwargs):
        """Calculates the generator's loss for fooling the discriminator."""
        _, fooling_logits = self.model(x_fake, kwargs.get('y_fake', None))
        fooling_labels = torch.ones_like(fooling_logits)
        return self.criterion(fooling_logits, fooling_labels)