import torch
from torch import nn, Tensor
from .mdm import MDM

from typing import (
    List
)

from argparse import Namespace
from model.motion_encoders import MotionEncoderAttentionBiGRU, STGCN
from model.wrappers import Discriminator, Critic
from model.utils import MLP
from model.utils.adversarial_utils import spectral_norm


class KineMIC(nn.Module):
    """
    Wrapper for 2-stream MDM with additional components
    """
    def __init__(self, model_args_dict, stream_names = ["prior", "target"], tau=0.07, judge=Namespace()):
        super(KineMIC, self).__init__()
        self.stream_names = stream_names
        assert stream_names == ["prior", "target"], "Will modify this, it's hardcoded for now"

        self.streams = nn.ModuleDict({
            'prior': MDM(**model_args_dict['prior'], stream='prior'),
            'target': MDM(**model_args_dict['target'], stream='target')
        })

        # the MIC module
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/tau)))
        self.latent_dim = model_args_dict['prior']['latent_dim']
        self.mic_module = MotionEncoderAttentionBiGRU(self.latent_dim, self.latent_dim, self.latent_dim)

        # Domain and Semantic Judges
        # Determine the type of judge
        Judge = Discriminator if judge.type == 'd' else Critic if judge.type == 'c' else None
        # init. judges
        if Judge is not None:
            if 'semantic' in judge.task:
                print("\nInitializing [Semantic] Judge...")
                self.semantic_judge = Judge(
                    model = MLP(
                        input_dim=self.latent_dim,
                        layer_dims=[256, 128],
                        dropout_prob=0.1
                    ),
                    out_dim=128,
                )
                if judge.reg == 'sn': # apply spectral norm
                    self.semantic_judge.apply(spectral_norm)
                print('> Size: %.3fM\n' % (sum(p.numel() for p in self.semantic_judge.parameters() if p.requires_grad) / 1000000.0))


            if 'style' in judge.task:
                print("\nInitializing [Style] Judge...")
                stgcn_model = STGCN(
                    graph_cfg=dict(layout='humanml', mode='stgcn_spatial'),
                    in_channels=3, # xyz input
                    num_stages=8, # 10
                    base_channels=32, # 64
                    ch_ratio=2, # 2
                    inflate_stages=[5, 8] # [5, 8]
                )
                self.style_judge = Judge(
                    model = stgcn_model,
                    out_dim= stgcn_model.out_channels,
                )
                if judge.reg == 'sn': # apply spectral norm
                    self.style_judge.apply(spectral_norm)
                print('> Size: %.3fM\n' % (sum(p.numel() for p in self.style_judge.parameters() if p.requires_grad) / 1000000.0))

        ###
        ### inner modules
        ###

        self.rot2xyz = self.streams['target'].rot2xyz
        self.encode_text = self.streams['prior'].encode_text


    def forward(self, x, timesteps, y=None, stream=None) -> Tensor:
        """Forward pass through the model for a specific stream"""
        assert stream is not None and y is not None, "Must specify stream and conditioning y for KineMIC"
        return self.streams[stream](x, timesteps, y=y)

    def share_common_parameters(self):
        """ Share frozen parameters across streams """
        prior_params = dict(self.streams['prior'].named_parameters())
        target_params = dict(self.streams['target'].named_parameters())
        for name, prior_param in prior_params.items():
            if name in target_params:
                target_param = target_params[name]
                if not prior_param.requires_grad and not target_param.requires_grad:
                    target_param = prior_param # weight tying

    def parameters_wo_clip(self) -> List[nn.Parameter]:
        """ Get parameters excluding CLIP model parameters """
        params = []
        for stream in self.stream_names:
            params.extend(self.streams[stream].parameters_wo_clip())
        return params

