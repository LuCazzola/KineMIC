"""
Code taken from PySKL ST-GCN implementation
> Added uniform sampling of motion clips
"""

import copy as cp
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from model.utils import mstcn, unit_gcn, unit_tcn
from model.utils.graph import Graph

EPS = 1e-4

class STGCNBlock(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class STGCN(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 clips_length=100,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs
        self.clips_length = clips_length # Store clips_length

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()
        
        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.out_channels = base_channels # base channels after last stage
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def _sample_uniform(self, x, lengths):
        """
        Uniformly samples a fixed-length clip from motion data.
        If the sequence is shorter than clips_length, it is padded with zeros.
        x: Tensor (N, M, T, V, C)
        lengths: actual sequence lengths (without padding)
        """
        N, M, T, V, C = x.shape
        device = x.device
        # 1. Calculate random start indices as before
        max_start = torch.clamp(lengths - self.clips_length, min=0).to(device)
        start_indices = (torch.rand(N, device=device) * max_start.float()).long()
        # 2. Create the ideal time indices (these may go out of bounds for short sequences)
        time_indices = torch.arange(self.clips_length, device=device).unsqueeze(0) + start_indices.unsqueeze(1)
        # 3. Create a mask of valid time steps (True where index is within the actual length)
        # Shape: (N, clips_length)
        mask = time_indices < lengths.unsqueeze(1)
        # 4. Clamp indices to be valid for gathering (this will repeat the last frame)
        clamped_indices = torch.clamp(time_indices, 0, T - 1)
        # 5. Gather data using the clamped indices
        idx = clamped_indices.view(N, 1, self.clips_length, 1, 1)
        expanded_idx = idx.expand(N, M, self.clips_length, V, C)
        x_sampled = torch.gather(x, 2, expanded_idx)
        # 6. Apply the mask to zero out the invalid, repeated frames
        # Reshape mask to (N, 1, clips_length, 1, 1) to broadcast correctly
        mask_expanded = mask.view(N, 1, self.clips_length, 1, 1)
        x_zero_padded = x_sampled * mask_expanded.to(x.dtype)
        return x_zero_padded

    def forward(self, x, y):
        
        # NOTE: while that's not really elegant, it's convenient,
        # especially because we're 100% sure that the correct lengths
        # are always provided, and noise from Synthetic motion or other
        # sources is avoided.
        x = self._sample_uniform(x, y['lengths'])

        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        return x