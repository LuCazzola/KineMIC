from torch.utils.data import DataLoader
from data_loaders.tensors import dual_collate
from data_loaders.tensors import t2m_collate, t2m_gt_collate_fn
from data_loaders.tensors import a2m_collate, a2m_gt_collate_fn, a2m_stgcn_evaluator_collate
from data_loaders.tensors import mixed_ta2m_collate, mixed_ta2m_gt_collate_fn, mixed_ta2m_evaluator_collate

def get_dataset_class(name):
    if name == 'humanml':
        from data_loaders.unified.data.dataset import HumanML3D
        return HumanML3D
    elif name == 'kit':
        from data_loaders.unified.data.dataset import KIT
        return KIT
    elif name == 'ntu60':
        from data_loaders.unified.data.dataset import NTU60
        return NTU60
    elif name == 'ntu120':
        from data_loaders.unified.data.dataset import NTU120
        return NTU120
    elif name == 'ntu-vibe':
        from data_loaders.unified.data.dataset import NTUVIBE
        return NTUVIBE
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(cond_mode='text', hml_mode='train', evaluator=None, batch_size=1):
    
    # Ground truth collate (applied during evaluation)
    if hml_mode == 'gt':
        if cond_mode == 'text':
            collate_fun = t2m_gt_collate_fn
        elif cond_mode == 'action':
            collate_fun = a2m_gt_collate_fn
        elif cond_mode == 'mixed':
            collate_fun = mixed_ta2m_gt_collate_fn
        else:
            raise ValueError('Not handled: hml_mode={}, cond_mode={}'.format(hml_mode, cond_mode))
    # Specific evaluators collate (for STGCN action recognition)
    elif evaluator is not None:
        if evaluator == 'stgcn':
            if cond_mode == 'action':
                collate_fun = a2m_stgcn_evaluator_collate
            elif cond_mode == 'mixed':
                collate_fun = mixed_ta2m_evaluator_collate
            else:
                raise ValueError('Not handled: hml_mode={}, cond_mode={}, evaluator={}'.format(hml_mode, cond_mode, evaluator))
        else:
            raise ValueError('Not handled: hml_mode={}, cond_mode={}'.format(hml_mode, cond_mode))
    # MDM training collate
    else:
        if cond_mode == 'text':
            collate_fun = t2m_collate
        elif cond_mode == 'action':
            collate_fun = a2m_collate
        elif cond_mode == 'mixed':
            collate_fun = mixed_ta2m_collate
        else:
            raise ValueError('Not handled: hml_mode={}, cond_mode={}'.format(hml_mode, cond_mode))

    return lambda x: collate_fun(x, batch_size)


def get_dataset(data_stream_args, split='train', mode='train', abs_path='.', device=None,
                data_rep=None, cond_mode=None, blacklist=set(), synth_data_folder=None): 
    '''
    Builds dataset object
    '''
    
    dataset_name = data_stream_args.dataset
    data_rep = data_rep if data_rep is not None else getattr(data_stream_args, 'data_rep', 'hml_vec')
    cond_mode = getattr(data_stream_args, 'cond_mode', 'text') # default to text if not specified

    dataset = get_dataset_class(dataset_name)(
        mode, split=split, abs_path=abs_path, device=device, data_stream_args=data_stream_args, 
        data_rep=data_rep, cond_mode=cond_mode, blacklist=blacklist, synth_data_folder=synth_data_folder
    )

    return dataset


def get_single_stream_dataloader(data_stream_args, batch_size, split='train', hml_mode='train', device=None,
                        evaluator=None, data_rep=None, oversample=None, blacklist=set(), synth_data_folder=None, shuffle=True):
    '''
    Build Dataloader given some data stream
    '''
    
    dataset = get_dataset(
        data_stream_args, split=split, mode=hml_mode, device=device,
        data_rep=data_rep, blacklist=blacklist, synth_data_folder=synth_data_folder
    )
    
    if oversample is not None and oversample > len(dataset):
        print(f'> Oversampling to: ', oversample)
        dataset.m_dataset.oversample(oversample)

    if len(dataset) < batch_size:
        print("> Reducing batch size to fit dataset ({} < {})".format(len(dataset), batch_size))
        batch_size = len(dataset)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=False,
        collate_fn=get_collate_fn(
            hml_mode=hml_mode,
            cond_mode=dataset.cond_mode,
            batch_size=batch_size, evaluator=evaluator
        )
    )

    return loader


def get_dual_stream_dataloader(prior_dataloader, target_dataloader, top_k_sp, shuffle=True):
    '''
    Loads a dual stream dataset
    '''
    from data_loaders.unified.data.dataset import DualMDMDataset
    dataset = DualMDMDataset(prior_dataloader.dataset.m_dataset, target_dataloader.dataset.m_dataset, top_k_sp)

    loader = DataLoader(
        dataset, batch_size=prior_dataloader.batch_size, shuffle=shuffle, num_workers=4, drop_last=False,
        collate_fn = lambda x, : dual_collate(x, target_dataloader.batch_size, target_dataloader.dataset.cond_mode)
    )
    return loader