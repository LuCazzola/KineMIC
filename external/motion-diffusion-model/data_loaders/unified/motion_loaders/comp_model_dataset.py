
import torch
import numpy as np

from tqdm import tqdm
from einops import rearrange
from abc import ABC, abstractmethod
from torch.utils.data import Dataset

from data_loaders.unified.scripts.motion_process import recover_from_ric
from utils import dist_util

class CompGeneratedMotionDataset(Dataset, ABC):
    """
    Samples motion using the provided model and diffusion process for eval. comparison
    """
    def __init__(self, args, model, diffusion, sample_fn, dataloader, mm_num_samples, mm_num_repeats, num_samples_limit, scale=1., clip_denoised=False):
        super().__init__()
        self.dataset = dataloader.dataset
        assert self.dataset.mode in ['eval'], 'Sanity check failed'

        # Determine how many samples to generate for specified dataloader
        real_num_batches = len(dataloader)
        limited_batches = num_samples_limit // dataloader.batch_size + 1
        is_full_eval_set = num_samples_limit == -1 or num_samples_limit is None
        if is_full_eval_set:
            real_num_batches = min(limited_batches, real_num_batches)
        effective_steps = real_num_batches if is_full_eval_set else limited_batches       

        # Repetitions per sample (for more robustness on eval)
        if mm_num_samples > 0:
            # Randomly pick how many repets to do on random batches
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size+1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            # one repeat for each batch
            mm_idxs = []

        generated_motion, mm_generated_motions = [], []
        model.eval()
        with torch.no_grad():
            for i, (motion, model_kwargs) in enumerate(tqdm(dataloader, total=effective_steps)):

                if not is_full_eval_set and len(generated_motion) >= num_samples_limit:
                    # collected enough samples
                    break
                
                model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}
                if scale != 1.: # add CFG scale to batch
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0], device=dist_util.dev()) * scale

                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                
                # Begin sampling
                mm_motions = []
                for rep in range(repeat_times):
                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    ) # (batch feats joints time)
                    
                    if rep == 0: # NOTE: reps are only to compute MutliModality metric
                        # Collect the generated motions
                        sub_dicts = [{
                            'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                            'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                        } for bs_i in range(dataloader.batch_size)]
                        self._update_data_dict(sub_dicts, model_kwargs, cap_len_calc=0)
                        generated_motion += sub_dicts

                    if is_mm:
                        # collect for MultiModality eval
                        for bs_i in range(dataloader.batch_size):
                            mm_motion = sample[bs_i].squeeze().permute(1, 0).cpu().numpy()
                            mm_motion = self.switch_to_eval_norm(mm_motion)
                            
                            mm_motions.append({
                                'motion': mm_motion,
                                'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                            })
                if is_mm:
                    mm_sub_dicts = [{
                        'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                    } for bs_i in range(dataloader.batch_size)]
                    self._update_data_dict(mm_sub_dicts, model_kwargs, cap_len_calc=1)
                    mm_generated_motions += mm_sub_dicts

        self.w_vectorizer = getattr(dataloader.dataset, 'w_vectorizer', None)
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions

    def switch_to_eval_norm(self, x):
        '''Re-Normalize with evaluator stats'''
        if self.dataset.mode != 'eval':
            # NOTE: gt dataset is by default normalized with evaluator stats
            # this class assumes dataset is either eval or gt mode
            return x

        # samples are stored in original dataset normalization, so de-normalize first
        x = self.dataset.de_normalize_motion(x)

        # Format convertion (if needed)
        if x.shape[-1] != self.dataset.mean_for_eval.shape[-1]: # model was trained on different feature set than the evaluator
            if x.shape[-1] == 263 and self.dataset.mean_for_eval.shape[-1] == 66: # FIXME hardcoded (xyz data rep)
                x = recover_from_ric(torch.tensor(x).float(), 22)
                x = rearrange(x, 'time joints xyz -> time (joints xyz)', xyz=3).numpy()
            else:
                raise ValueError('Cannot convert motion of shape {} into {} data rep.'.format(x.shape, self.dataset.data_rep))

        # re-normalize with eval stats
        x = (x - self.dataset.mean_for_eval) / self.dataset.std_for_eval

        return x

    @abstractmethod
    def _update_data_dict(self, data_dict, model_kwargs, cap_len_calc=0):
        """
        Update the given data dict with conditioning keys.
        Does nothing by default, providing only motion-related keys.
        """
        pass
        
    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length = data['motion'], data['length']
        motion = self.switch_to_eval_norm(motion)
        return motion, m_length


class CompGeneratedDatasetT2M(CompGeneratedMotionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_data_dict(self, data_dict, model_kwargs, cap_len_calc=0):
        '''update for T2M data dict'''
        tokens = [t.split('_') for t in model_kwargs['y']['tokens']] # NOTE: calling this here creates tiny ovehead, but it's clean
        
        for bs_i in range(len(data_dict)):
            data_dict[bs_i]['caption'] = model_kwargs['y']['text'][bs_i]
            data_dict[bs_i]['tokens'] = tokens[bs_i]
            
            if cap_len_calc == 0:
                data_dict[bs_i]['cap_len'] = tokens[bs_i].index('eos/OTHER') + 1
            elif cap_len_calc == 1:
                data_dict[bs_i]['cap_len'] = len(tokens[bs_i])
            else:
                raise ValueError('Unknown cap_len_calc value: {}'.format(cap_len_calc))

    def __getitem__(self, item):
        
        motion, m_length = super().__getitem__(item)

        data = self.generated_motion[item]
        caption, tokens = data['caption'], data['tokens']
        sent_len = data['cap_len']

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token] # type: ignore
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompGeneratedDatasetA2M(CompGeneratedMotionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_data_dict(self, data_dict, model_kwargs, cap_len_calc=0):
        '''update for A2M data dict'''
        for bs_i in range(len(data_dict)):
            data_dict[bs_i]['action'] = model_kwargs['y']['action'][bs_i]
            data_dict[bs_i]['action_text'] = model_kwargs['y']['action_text'][bs_i]

    def __getitem__(self, item):
        motion, m_length = super().__getitem__(item)
        data = self.generated_motion[item]
        action, action_text = data['action'], data['action_text']
        return motion, m_length, action, action_text


# Wrapper (defines interface and calls appropriate dataset class)
class CompMDMGeneratedDataset(Dataset):
    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
        super().__init__()

        self.args = args
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.model = model
        self.cond_mode = self.dataset.cond_mode
        self.max_motion_length = max_motion_length # unused, but kept for compatibility
        assert mm_num_samples < len(dataloader.dataset)
        
        # NOTE - hardcoded
        USE_DDIM = False
        CLIP_DENOISED = False
        
        sample_fn = (
            diffusion.p_sample_loop if not USE_DDIM else diffusion.ddim_sample_loop
        )

        if self.cond_mode == 'text':
            self.dataset = CompGeneratedDatasetT2M(args, model, diffusion, sample_fn, dataloader, mm_num_samples, mm_num_repeats, num_samples_limit, scale, clip_denoised=CLIP_DENOISED)
        elif self.cond_mode == 'action':
            self.dataset = CompGeneratedDatasetA2M(args, model, diffusion, sample_fn, dataloader, mm_num_samples, mm_num_repeats, num_samples_limit, scale, clip_denoised=CLIP_DENOISED)
        elif self.cond_mode == 'mixed':
            # NOTE: in this work, when using a mixed conditioning model, evaluation is still done only on action-to-motion
            self.dataset = CompGeneratedDatasetA2M(args, model, diffusion, sample_fn, dataloader, mm_num_samples, mm_num_repeats, num_samples_limit, scale, clip_denoised=CLIP_DENOISED)
        else:
            raise ValueError('Unknown conditioning mode: {}'.format(self.cond_mode))
    
        self.mm_generated_motion = self.dataset.mm_generated_motion

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        return self.dataset[item]
        