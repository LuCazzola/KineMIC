import torch
from einops import rearrange
from torch.utils.data._utils.collate import default_collate
from data_loaders.unified.scripts.motion_process import recover_from_ric

from data_loaders.unified.data.dataset import DualMDMDataset

def unwrap_batch(batch, dataset, active_streams):
    if isinstance(dataset, DualMDMDataset):
        (prior_motion, prior_cond), (target_motion, target_cond) = batch
        motion = {'prior': prior_motion, 'target': target_motion}
        cond = {'prior': prior_cond, 'target': target_cond}
    else:
        assert len(active_streams) == 1, "In single-stream datasets, only one active stream should be present."
        motion = {active_streams[0]: batch[0]}
        cond = {active_streams[0]: batch[1]}
    return motion, cond

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    
def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text'] for b in notnone_batches]
        cond['y'].update({'action_text': action_text})
    
    if 'tag' in notnone_batches[0]:
        tagbatch = [b['tag'] for b in notnone_batches]
        cond['y'].update({'tag': tagbatch})

    if 'key' in notnone_batches[0]:
        cond['y'].update({'db_key': [b['key'] for b in notnone_batches]})

    return motion, cond
##
### Utility
##

def adapt_batch(batch, target_batch_size):
    repeat_factor = -(-target_batch_size // len(batch))  # Ceiling division
    repeated_batch = batch * repeat_factor 
    return repeated_batch[:target_batch_size]

##
### T2M Collate
##

def t2m_collate(batch, target_batch_size):
    full_batch = adapt_batch(batch, target_batch_size)
    adapted_batch = [{
            'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': b[2],
            'tokens': b[6],
            'lengths': b[5],
            'key': b[7] if len(b) > 7 else None,
        } for b in full_batch]
    return collate(adapted_batch)

def t2m_gt_collate_fn(batch, *kargs):
    batch.sort(key=lambda x: x[3], reverse=True) # sort by caption length
    return default_collate(batch)

##
### A2M Eval.
##

def a2m_collate(batch, target_batch_size):
    full_batch = adapt_batch(batch, target_batch_size)
    adapted_batch = [{
        'inp': torch.tensor(b[0].T).float().unsqueeze(1) if b[0] is not None else None, # [T, D] -> [D, 1, T]
        'lengths': b[1],
        'action': b[2],
        'action_text': b[3],
        'key': b[4] if len(b) > 4 else None,
    } for b in full_batch]
    return collate(adapted_batch)

def a2m_stgcn_evaluator_collate(batch, target_batch_size):
    '''Assumes STGCN evaluator'''
    full_batch = adapt_batch(batch, target_batch_size)
    # (motion, lengths, action, action_text)
    adapted_batch = [{
        'inp': rearrange(torch.tensor(b[0]).float(), 't (j xyz) -> t j xyz ', xyz=3).unsqueeze(0), # [T, J*3] -> [1, T, J, 3]
        'lengths': b[1],
        'action': b[2],
        'action_text': b[3],
        'key': b[4] if len(b) > 4 else None,
    } for b in full_batch]
    return collate(adapted_batch)

def a2m_gt_collate_fn(batch, *kargs):
    batch = default_collate(batch)
    # batch : (motion, lengths, action, action_text)
    # motion: [B, T, D], with D = J*3 or 263 (hml_vec)
    if batch: # format for STGCN evaluator
        batch[0] = rearrange(batch[0], 'batch time (joints xyz) -> batch time joints xyz ', xyz=3)
        batch[0] = batch[0].unsqueeze(1).float() # adding skeleton dimension
    
    return batch

def a2m_mm_collate_fn(batch, *kargs): # FIXME: needs testing
    batch = default_collate(batch)
    # (motion, lengths)
    # input motion motion : (batch reps time dim), dim depends on the format used
    if batch:
        if batch[0].shape[-1] == 263: # a hml_vec format was given
            batch[0] = recover_from_ric(batch[0], 22) # FIXME: hardcoded 22 joints (smpl-nohands)
            batch[0] = rearrange(
                batch[0].float(),
                f'batch reps time joints xyz -> (batch reps) time joints xyz',
                xyz=3
            )
        else:
            batch[0] = rearrange(
                batch[0].float(),
                f'batch reps time (joints xyz) -> (batch reps) time joints xyz',
                xyz=3
            )
        batch[0] = batch[0].unsqueeze(1) # adding skeleton dimension
        batch[1] = batch[1].squeeze(0) # remove redundant dim
    
    return batch

##
## mixed Text-Action-2-Motion
##

def mixed_ta2m_collate(batch, target_batch_size):
    full_batch = adapt_batch(batch, target_batch_size)
    adapted_batch = [{
            'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': b[2],
            'tokens': b[6],
            'lengths': b[5],
            'action': b[7],
            'action_text': b[8],
            'tag': b[9],
            'key': b[10] if len(b) > 10 else None,
        } for b in full_batch]
    return collate(adapted_batch)

# NOTE: evaluators still use the a2m collate
# So we trash the Text-related components and
# call the relative a2m collate functions

def ta2m_to_a2m(batch):
    return [((b[4], b[5], b[7], b[8], b[10]) if len(b) > 10 else (b[4], b[5], b[7], b[8])
            ) if b is not None else None
        for b in batch]
    
def mixed_ta2m_evaluator_collate(batch, *kargs):
    return a2m_stgcn_evaluator_collate(ta2m_to_a2m(batch), *kargs)

def mixed_ta2m_gt_collate_fn(batch, *kargs):
    return a2m_gt_collate_fn(ta2m_to_a2m(batch), *kargs)

def mixed_ta2m_mm_collate_fn(batch, *kargs):
    return a2m_mm_collate_fn(ta2m_to_a2m(batch), *kargs)

##
## DUAL collate (DualMDMDataset)
##

def dual_collate(batch, target_batch_size, target_cond_mode):
    prior_data, target_data = zip(*batch) # unzip
    # parse through respective collate
    prior_data = t2m_collate(prior_data, target_batch_size)

    if target_cond_mode == 'action':
        target_data = a2m_collate(target_data, target_batch_size)
    elif target_cond_mode == 'text':
        target_data = t2m_collate(target_data, target_batch_size)
    elif target_cond_mode == 'mixed':
        target_data = mixed_ta2m_collate(target_data, target_batch_size)
    else:
        raise ValueError('Not handled: target_cond_mode={}'.format(target_cond_mode))
    
    # reuturn both
    return prior_data, target_data