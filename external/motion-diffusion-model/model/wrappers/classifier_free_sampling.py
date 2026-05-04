import torch.nn as nn
from copy import deepcopy
from utils.misc import wrapped_getattr
from typing import Union

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode
        if self.cond_mode == 'text':
            self.encode_text = self.model.encode_text

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action', 'mixed']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out, _ = self.model(x, timesteps, y)
        out_uncond, _ = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond)), {} # second return value is a placeholder (consistency with MDM output) 

    def __getattr__(self, name, default=None):
        # this method is reached only if name is not in self.__dict__.
        return wrapped_getattr(self, name, default=None)


def wrap_w_classifier_free_sampling(model):
    """
    Wraps a model with ClassifierFreeSampleModel for classifier-free guidance sampling.
    """
    from model import MDM, KineMIC

    if isinstance(model, (MDM)):
        return ClassifierFreeSampleModel(model)
    elif isinstance(model, (KineMIC)):
        # wrap target stream only in case of KineMIC
        return ClassifierFreeSampleModel(model.streams['target'])

    raise TypeError(f"Unsupported model type: {type(model)}")