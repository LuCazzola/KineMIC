import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from data_loaders.unified.motion_loaders.comp_model_dataset import CompMDMGeneratedDataset
from data_loaders.tensors import t2m_gt_collate_fn, a2m_gt_collate_fn, a2m_mm_collate_fn

class MMGeneratedDataset(Dataset):
    """Dataloader for MultiModal eval."""
    def __init__(self, opt, motion_dataset):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            motion = motion[None, :]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=np.int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()

        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens


def get_mdm_loader(args, model, diffusion, batch_size, ground_truth_loader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale, cond_mode):
    opt = {
        'name': 'test',  # FIXME what the use for that? lol
    }
    print('Generating %s ...' % opt['name'])

    # When CompMDMGeneratedDataset is instantiated, the generation process is triggered
    # and the generated motions are stored in the dataset object
    dataset = CompMDMGeneratedDataset(args, model, diffusion, ground_truth_loader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale)
    # MMGeneratedDataset simply wraps around the generated motions
    # and specifically process motion for MultiModality metric
    mm_dataset = MMGeneratedDataset(opt, dataset)

    # The below dataloaders instead, are called within the evaluation script
    # e.g. ntu_eval.py in out case

    if cond_mode in ['text']:
        m_collate = t2m_gt_collate_fn
    elif cond_mode in ['action', 'mixed']:
        # NOTE: mixed datasets still need to be evaluated with a2m collate
        # e.g. for NTU RGB we only care about associated actions, not to the
        # text that might have been employed for samples' generation
        m_collate = a2m_gt_collate_fn
    else:
        raise ValueError('Not handled: cond_mode={}'.format(cond_mode))

    # NOTE: bs must not be changed! this will cause a bug in R precision calc! NOTE (old comment there since fork, still an issue? idk)
    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, collate_fn=m_collate)
    
    if cond_mode in ['text']:
        mm_collate = default_collate
    elif cond_mode in ['action', 'mixed']:
        mm_collate = a2m_mm_collate_fn
    else:
        raise ValueError('Not handled: cond_mode={}'.format(cond_mode))

    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0, collate_fn=mm_collate)
    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader
