import os
import json
from os.path import join as pjoin

def get_ntu_blacklist(dataset, fewshot_id, task_split, avail_splits=['train', 'val']):

    blacklist = {split: [] for split in avail_splits}
    whitelist = {split: [] for split in avail_splits}

    dataset = dataset.upper()

    if fewshot_id is not None and fewshot_id != 'full':
        # If a few-shot ID is provided, create a blacklist of samples not belonging to the few-shot classes
        subset_path = pjoin('.', 'dataset', dataset, 'splits', 'fewshot', fewshot_id)
        assert os.path.exists(subset_path), 'Few-shot split ID [{}] not found in {}'.format(fewshot_id, subset_path)
        with open(pjoin(subset_path, 'meta.json'), 'r') as f:
            class_list = json.load(f)['class_list'] # class ids in the few-shot split

        data_path = pjoin('.', 'dataset', dataset, 'splits', 'default', task_split)
        for split in avail_splits:
            with open(pjoin(data_path, f'{split}_y.txt'), 'r') as f: # all class ids in the split
                classes = [int(line.strip()) for line in f.readlines()]
            with open(pjoin(data_path, f'{split}.txt'), 'r') as f: # all sample ids in the split
                samples = [line.strip() for line in f.readlines()]
            blacklist[split] += [s for s, c in zip(samples, classes) if c not in class_list] # blacklist samples not in the few-shot class list
            whitelist[split] += [s for s, c in zip(samples, classes) if c in class_list]

    for split in avail_splits:
        blacklist[split] = set(blacklist[split])
        whitelist[split] = set(whitelist[split])

    return blacklist, whitelist

def get_ntu_samples_in_split(dataset, split, task_split, fewshot_id=None):

    dataset = dataset.upper()

    if fewshot_id is None:
        data_path = pjoin('.', 'dataset', dataset, 'splits', 'default', task_split)
    else:
        data_path = pjoin('.', 'dataset', dataset, 'splits', 'fewshot', fewshot_id, task_split)
    
    with open(pjoin(data_path, f'{split}.txt'), 'r') as f:
        samples = [line.strip() for line in f.readlines()]
    return samples