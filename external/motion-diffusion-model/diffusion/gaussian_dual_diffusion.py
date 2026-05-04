from time import thread_time
import numpy as np
import torch as th
from torch import nn
from copy import deepcopy
from .gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType, _extract_into_tensor

from utils.loss_util import contrastive_loss, soft_nearest_neighbors_loss, entropy_regularization
from einops import rearrange
from data_loaders.unified.scripts.motion_process import recover_from_ric

from model.utils.adversarial_utils import gradient_penalty

from utils import dist_util
from torch import Tensor

# Typing options
from data_loaders.unified.data.dataset import DualMDMDataset
from contextlib import nullcontext
from argparse import Namespace
from model import KineMIC
from typing_extensions import Literal
from typing import (
    Optional,
    Dict,
    List,
    Callable,
)

ModelKwargs = Dict[str, Dict[Literal['y'], Dict[str, Tensor]]]

class GaussianDualDiffusion(GaussianDiffusion):
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        lambd : Namespace,
        rescale_timesteps=False,
        data_rep='rot6d',
        stream_warmup_steps=0,
        data={},
        bank_limit=60,
        pretrain_diffusion_steps=50, # number of diffusion steps used in the pretrained prior model
        dww=False, # dynamic window weighting
        **kargs,
    ):
        super().__init__(betas=betas, model_mean_type=model_mean_type, model_var_type=model_var_type, loss_type=loss_type, lambd=lambd, rescale_timesteps=rescale_timesteps, data_rep=data_rep, **kargs)
        self.lambd = lambd
        self.stream_warmup_steps = stream_warmup_steps
        self.iteration = 0
        self.data = data
        self.dww = dww
        self.bank = {'x': None, 'action': None}
        self.bank_limit = bank_limit
        self.PRIOR_NUM_TIMESTEPS = pretrain_diffusion_steps

        if self.lambd.adversarial > 0.:
            self.judge_opt = {'semantic': None, 'style': None} # to be set from outside

    def training_losses(
        self,
        model : KineMIC,
        x_start : Dict[str, Tensor],
        t : Tensor,
        model_kwargs : Optional[ModelKwargs] = {},
        noise : Optional[Tensor] = None,
        dataset : Optional[DualMDMDataset] = None
    ):
        assert list(model_kwargs.keys()) == model.stream_names
        assert model_kwargs is not None, "model_kwargs must be provided for KineMIC"
        assert x_start['prior'].shape == x_start['target'].shape, "Input tensors must have the same shape"

        mask = {
            stream : model_kwargs[stream]['y']['mask']
            for stream in model.stream_names
        }

        if noise is None:
            noise = th.randn_like(x_start['target'])
    
        x_t = {
            stream : self.q_sample(x_start[stream], t, noise=noise)
            for stream in model.stream_names
        }
        terms = {}
        model_output, out_dict = {}, {}

        ##
        ## Begin Processing
        ##

        with th.no_grad():
            _, out_dict['prior'] = model(x_t['prior'], self._scale_prior_timesteps(t), stream='prior', **model_kwargs['prior'])
        model_output['target'], out_dict['target'] = model(x_t['target'], self._scale_timesteps(t), stream='target', **model_kwargs['target'])

        # --- Target Reconstruction Loss ---
        target = {
            ModelMeanType.START_X: x_start['target'],
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        terms["target_rec"] = self.masked_lx(
            target, model_output['target'],
            mask['target'], loss_fn=self.diff_loss
        )

        # --- Contrastive Loss ---
        # > Soft Nearest Neighbors Loss with Memory Bank
        # > Samples belonging to the same action class should have similar latent representations (in the MIC latent space)
        if self.lambd.contrastive > 0.:
            # Get latent representations from the MIC module
            z_prior, z_prior_attention_weights = model.mic_module(
                out_dict['prior']['latent'].detach(), model_kwargs['prior']['y']['lengths']
            )
            z_target, _ = model.mic_module(
                out_dict['target']['latent'], model_kwargs['target']['y']['lengths']
            )
            # Get pseudo labelling and update memory bank 
            prior_actions = target_actions = model_kwargs['target']['y']['action'].squeeze()
            z_prior_bank, prior_actions_bank = self.update_bank(z_prior, target_actions)
            # concat current computation with the memory bank
            if z_prior_bank is not None and prior_actions_bank is not None : # (empty bank check)
                z_prior = th.cat([z_prior, z_prior_bank], dim=0)
                prior_actions = th.cat([prior_actions, prior_actions_bank], dim=0)
            # .
            model.logit_scale.data = th.clamp(model.logit_scale.data, 0, 4.6052) # clamping learnable logit scale
            terms['contrastive'] =  soft_nearest_neighbors_loss(
                z_target, z_prior,
                target_actions, prior_actions,
                model.logit_scale,            
            )
        
        if self.lambd.window_rec > 0. or self.lambd.window_distill > 0. or self.lambd.adversarial > 0.:
            
            # NOTE: this is very important, we do not want gradients to flow
            # through the MIC module when processing the prior stream again
            z_prior_attention_weights = z_prior_attention_weights.detach()
            
            # For each item, identify the sequence of "target lenght" within
            # the prior motion that has the highest cumulative attention weight.
            window_start_indices, window_lengths = self._find_best_window_starts(
                z_prior_attention_weights,
                model_kwargs['prior']['y']['lengths'],
                model_kwargs['target']['y']['lengths']
            )

            for i, texts in enumerate(model_kwargs['prior']['y']['text']):
                print(f"Sample {i} - {model_kwargs['prior']['y']['db_key'][i]}:")
                print(f"  Caption: {texts}")
                print(f"  Action Match: {model_kwargs['target']['y']['action_text'][i]}")
                print(f"  Window Start: {window_start_indices[i].item()}, Length: {window_lengths[i].item()}")
            print("==="*30)
            # Extract the window as a new tensor (both x_start and the noise)
            x_start_prior_window = self._align_tensor_to_window(
                x_start['prior'], window_start_indices, window_lengths
            )
            noise_window = self._align_tensor_to_window(
                noise, window_start_indices, window_lengths
            )
            # Get the noised window
            x_t_prior_window = self.q_sample(x_start_prior_window, t, noise=noise_window)
            # Parse the window through target model. The "prior window" becomes a new training sample
            # for the target stream, and the target sample's conditioning is used as pseudo-label for it
            prior_window_prediction, prior_window_out_dict = model(
                x_t_prior_window, self._scale_timesteps(t), stream='target', **model_kwargs['target']
            )

            # This block computes the weights (dww) but does not affect the gradient flow for the main model
            if self.dww :
                with th.no_grad():
                    # Get latent rep for the processed prior window.
                    z_prior_window, _ = model.mic_module(
                        prior_window_out_dict['latent'], window_lengths
                    )
                    z_target_norm = th.nn.functional.normalize(z_target.detach(), p=2, dim=1)
                    z_prior_window_norm = th.nn.functional.normalize(z_prior_window, p=2, dim=1)
                    cosine_sim = th.sum(z_target_norm * z_prior_window_norm, dim=1)
                    dynamic_window_weighting = (cosine_sim + 1.0) / 2.0 # from [-1, 1] to [0, 1] range
                    # For monitoring
                    terms['dww_mean'] = dynamic_window_weighting.mean().item()
                    terms['dww_std'] = dynamic_window_weighting.std().item()
                    terms['dww_median'] = dynamic_window_weighting.median().item()
            else:
                dynamic_window_weighting = 1.0
            
            # --- Window Reconstruction Loss ---
            if self.lambd.window_rec > 0.:
                window_target = {
                    ModelMeanType.START_X: x_start_prior_window,
                    ModelMeanType.EPSILON: noise_window,
                }[self.model_mean_type]
                terms['window_rec'] = self.masked_lx(
                    window_target, prior_window_prediction, mask['target'], loss_fn=self.diff_loss
                )
            
            # --- Window Distillation Loss ---
            # > Between the prior & target last transformer block representations w.r.t. the identified windows 
            if self.lambd.window_distill > 0.:
                student_window_feats = prior_window_out_dict['latent'].permute(1, 2, 0) # (T, B, D) -> (B, D, T)
                teacher_window_feats = self._align_tensor_to_window(
                    out_dict['prior']['latent'].permute(1, 2, 0), 
                    window_start_indices, window_lengths,
                )
                # .
                terms['window_distill'] = self.masked_lx(
                    teacher_window_feats.detach(), student_window_feats, mask['target'], loss_fn=self.diff_loss
                )

            # --- Adversarial Loss ---
            # N.B. (Unused in the paper)
            if self.lambd.adversarial > 0. and self.stream_warmup_steps <= self.iteration: 
                
                # --- Semantic Adversarial Loss ---
                # > On the Latent space of the MIC module
                # > The processed prior window latent rep. should be indistinguishable from the target motion latent rep.
                if self.judge_opt['semantic'] is not None:

                    z_real = z_target
                    with th.no_grad():
                        # NOTE: the adversarial loss should not interfere with
                        # gradients of the MIC module
                        z_fake, _ = model.mic_module(
                            prior_window_out_dict['latent'], window_lengths
                        )
                    # .
                    terms['semantic_d'], terms['semantic_g'] = self.adversarial_step(
                        model.semantic_judge, 'semantic',
                        z_real, z_fake,
                    )
                    
                # --- Style Adversarial Loss ---
                # > On the reconstructed motion sequences
                # > Prior Window reconstruction should be indistinguishable from the real target motion
                if self.judge_opt['style'] is not None:
                    # the reconstructed target motion sequence (in xyz space)
                    x_start_real = dataset.de_normalize_motion(x_start['target'], gpu=True)
                    x0hat_real = dataset.de_normalize_motion(model_output['target'], gpu=True)
                    x0hat_fake = dataset.de_normalize_motion(prior_window_prediction, gpu=True)
                    # Extract xyz
                    x_start_real = recover_from_ric(rearrange(x_start_real, 'b c j t -> b t (j c)'), 22).unsqueeze(1) 
                    x0hat_real = recover_from_ric(rearrange(x0hat_real, 'b c j t -> b t (j c)'), 22).unsqueeze(1) # unsqueeze to add number of skeletons dim.
                    x0hat_fake = recover_from_ric(rearrange(x0hat_fake, 'b c j t -> b t (j c)'), 22).unsqueeze(1) # unsqueeze to add number of skeletons dim.
                    # Concatenate Gt and Gen. outputs along the batch dimension
                    # We essentially consider as real samples both predictions and GT
                    x_real = th.cat([x_start_real, x0hat_real], dim=0)
                    real_lengths = th.cat([model_kwargs['target']['y']['lengths'], model_kwargs['target']['y']['lengths']], dim=0)
                    x_fake = th.cat([x0hat_fake, x0hat_fake], dim=0)  # duplicate to match the real samples count
                    fake_lengths = th.cat([window_lengths, window_lengths], dim=0)
                    # .
                    terms['style_d'], terms['style_g'] = self.adversarial_step(
                        model.style_judge, 'style',
                        x_real, x_fake,
                        **{'y_real': {'lengths': real_lengths},
                           'y_fake': {'lengths': fake_lengths}
                        }
                    )

        weighted_window_rec = terms.get('window_rec', 0.) * dynamic_window_weighting
        weighted_window_distill = terms.get('window_distill', 0.) * dynamic_window_weighting

        terms["loss"] = ( self.lambd.rec * terms.get('target_rec', 0.) ) + \
                        ( self.lambd.contrastive * terms.get('contrastive', 0.) ) + \
                        ( self.lambd.window_rec * weighted_window_rec ) + \
                        ( self.lambd.window_distill * weighted_window_distill ) + \
                        ( self.lambd.adversarial * ( \
                            (self.lambd.adversarial_alpha * terms.get('style_g', 0.)) + \
                            (1-self.lambd.adversarial_alpha) * terms.get('semantic_g', 0.))
                        )

        return terms
    
    # ... (the rest of the helper methods remain unchanged) ...
    def update_bank(self, prior_embeddings, prior_actions):
        """
        Memory bank for prior embeddings and corresponding action labels.
        returns : current bank state
        """
        bank_x, bank_action = self.bank['x'], self.bank['action'] 
        embeddings, actions = prior_embeddings.detach(), prior_actions.detach()

        if bank_x is None and bank_action is None:
            # init.
            self.bank['x'] = embeddings
            self.bank['action'] = actions
        else:
            # update
            self.bank['x'] = th.cat([self.bank['x'], embeddings], dim=0)
            self.bank['action'] = th.cat([self.bank['action'], actions], dim=0)

        if self.bank['x'].shape[0] > self.bank_limit:
            # pop old
            self.bank['x'] = self.bank['x'][-self.bank_limit:]
            self.bank['action'] = self.bank['action'][-self.bank_limit:]
        
        return bank_x, bank_action
    

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def _scale_prior_timesteps(self, t):
        t = th.tensor(th.ceil(t * (self.PRIOR_NUM_TIMESTEPS / self.num_timesteps)), dtype=t.dtype)
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.PRIOR_NUM_TIMESTEPS)
        return t
    
    def _find_best_window_starts(self, scores, input_lengths, window_lengths):
        """
        For each item in a batch, finds the start index of a sliding window
        of a given size that maximizes the sum of given scores within it.
        """
        batch_size, max_seq_len = scores.shape
        # Create a mask to zero out scores beyond the valid prior_lengths.
        arange_seq = th.arange(max_seq_len, device=scores.device).expand(batch_size, -1)
        prior_mask = arange_seq < input_lengths.unsqueeze(1)
        masked_attention = scores * prior_mask
        # Compute the cumulative sum to efficiently calculate all sliding window sums.
        summed_attention = th.nn.functional.pad(masked_attention.cumsum(dim=1), (1, 0))
        # Create indices for the end of each possible window.
        window_end_indices = arange_seq + window_lengths.unsqueeze(1)
        # Gather the cumulative sums at the start and end of each potential window.
        end_sums = summed_attention.gather(1, window_end_indices.clamp(max=max_seq_len))
        start_sums = summed_attention.gather(1, arange_seq)  # Starts are simply at indices 0, 1, ...
        # Calculate the score for every possible window start by subtracting cumulative sums.
        window_scores = end_sums - start_sums
        # Invalidate scores for windows that start too late to be valid.
        invalid_starts_mask = (arange_seq + window_lengths.unsqueeze(1)) > input_lengths.unsqueeze(1)
        window_scores.masked_fill_(invalid_starts_mask, -float('inf'))        
        # Find the index of the maximum score for each item in the batch.
        best_window_starts = window_scores.argmax(dim=1)
        effective_window_lengths = th.min(input_lengths, window_lengths) # edge case when target length > prior length, never happens in practice
        # .
        return best_window_starts, effective_window_lengths

    def _align_tensor_to_window(self, x: Tensor, window_start: Tensor, window_len: Tensor):
        """
        Creates a new tensor of the same size as x.
        For each item in the batch, a window of length window_len[i] is moved
        to the beginning, and the rest is padded with the window's last value.
        """
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions"
        assert x.shape[0] == window_start.shape[0] == window_len.shape[0], "Batch dimensions must match"
        B, T = x.shape[0], x.shape[-1]
        assert th.all(window_len > 0), "Window length cannot be zero"
        assert th.all((window_start + window_len) <= T), "Window cannot go out of bounds"

        # Create the full index map in 2D (Batch, Time)
        time_indices = th.arange(T, device=x.device).expand(B, -1)
        mask = time_indices < window_len.unsqueeze(1)
        window_indices = window_start.unsqueeze(1) + time_indices
        padding_indices = (window_start + window_len - 1).unsqueeze(1)
        # Select indices based on the mask and clamp them to the valid range [0, T-1]
        prior_indices = th.where(mask, window_indices, padding_indices).clamp(0, T - 1)
        # Reshape and expand indices to match the N-D shape of the input tensor
        view_shape = (B,) + (1,) * (x.dim() - 2) + (T,)
        expanded_indices = prior_indices.view(view_shape).expand(x.shape)
        # Gather the data using the computed indices
        return th.gather(x, dim=-1, index=expanded_indices)
    
    def adversarial_step(self, judge, name, z_real, z_fake, **kwargs):
        """
        Performs one step of adversarial training for the given judge (discriminator/critic).
        """
        # Phase 1: Train the Discriminator
        self.judge_opt[name].zero_grad()
        d_loss = judge.get_d_loss(z_real, z_fake, **kwargs)
        
        if self.lambd.gp > 0. : # if using gradient penalty, apply it
            gp = gradient_penalty(judge, z_real, z_fake, kwargs.get('y_real', None))
            d_loss = d_loss + self.lambd.gp * gp
        
        d_loss.backward()
        self.judge_opt[name].step()

        # Phase 2: Train the Target Model (Generator) to fool the Discriminator
        for p in judge.parameters():
            p.requires_grad = False
        g_loss = judge.get_g_loss(z_fake, **kwargs)
        for p in judge.parameters():
            p.requires_grad = True

        return d_loss.item(), g_loss