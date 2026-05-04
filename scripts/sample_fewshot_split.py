import os
import argparse
import random
from types import SimpleNamespace
import numpy as np
from itertools import chain
import json
import pickle
from tqdm import tqdm
from os.path import join as pjoin

from scripts.ntu_preproc import filter_data_consistency
from utils.constants import DATA_FILENAME, DATA_NUM_CLASSES, DATA_IGNORE_CLASSES, SKEL_INFO
from utils.humanml3d import cal_mean_variance


def parse_split_files(split_dir, split_sets):
    """Find all existing split .txt files under each split name and set."""
    split_names = [f for f in os.listdir(split_dir) if os.path.isdir(pjoin(split_dir, f))]
    split_files = [
        pjoin(split_dir, s_name, f"{s_set}.txt")
        for s_name in split_names for s_set in split_sets
        if os.path.exists(pjoin(split_dir, s_name, f"{s_set}.txt"))
    ]
    return split_files, split_names


def process_split_file(split_file, class_list, N, annotations_path, out_dir, outliers, with_stats=False):
    """Processes one split file: sampling, writing, and computing stats."""
    with open(split_file, 'r') as f:
        sample = [line.strip() for line in f]

    label_path = split_file.replace('.txt', '_y.txt')
    assert os.path.exists(label_path), f"Missing labels for {split_file}"
    with open(label_path, 'r') as f:
        label = [int(line.strip()) for line in f]

    for o in outliers:
        if o in sample:
            idx = sample.index(o)
            print(f"\t\t[Outlier Removal] {o} from class {label[idx]}")
            sample.pop(idx)
            label.pop(idx)

    # Group samples by class
    class_sample = [[] for _ in class_list]
    for idx, s in enumerate(sample):
        if label[idx] not in class_list:
            continue
        class_idx = class_list.index(label[idx])
        class_sample[class_idx].append(s)

    assert all(len(s) >= N for s in class_sample), "Some classes have fewer samples than requested."
    
    # NOTE: we take N shots only from 'train' split.
    # that's because val/test should be invariant to the choice of training samples.
    low_resource_data = [random.sample(s, N) for s in class_sample] if 'train' in split_file else class_sample

    # save split + labels
    split_name = os.path.splitext(os.path.basename(split_file))[0]
    with open(pjoin(out_dir, f"{split_name}.txt"), 'w') as f:
        for s in low_resource_data:
            f.write('\n'.join(s) + '\n')
    with open(pjoin(out_dir, f"{split_name}_y.txt"), 'w') as f:
        for class_idx, s in enumerate(low_resource_data):
            f.write('\n'.join([str(class_list[class_idx])] * len(s)) + '\n')

    filenames = list(chain.from_iterable(low_resource_data))
    print(f"\t\t[{split_name}] sampled {len(filenames)}...")
    if with_stats:
        # compute Mean-Std w.r.t. current Few-Shot training split
        for attr in ['joint_vecs', 'joints']:
            print(f"\t\tComputing [{attr}] stats...")
            d_path = getattr(annotations_path, attr)
            all_motion = [np.load(pjoin(d_path, fn + '.npy')) for fn in tqdm(filenames, desc='\tLoading sampled motions')]
            motion = np.concatenate(all_motion, axis=0)
            if attr == 'joint_vecs':
                # for hml vec. representation
                mean, std = cal_mean_variance(motion, SKEL_INFO["smpl"].joints_num)
            else :
                # for general representation (e.g. xyz, rot)
                mean = np.mean(motion, axis=0)
                std = np.std(motion, axis=0)

            np.save(pjoin(out_dir, f'Mean_{attr}.npy'), mean)
            np.save(pjoin(out_dir, f'Std_{attr}.npy'), std)
    else:
        print("\t\tSkipping stats computation.")


def merge_split(data, split_names, out_path):
    """
    Merge few-shot splits with the full dataset.
    Keeps full entries for non-fewshot classes, and trims fewshot classes to sampled entries.
    """
    with open(pjoin(out_path, 'meta.json'), 'r') as f:
        class_list = set(json.load(f)['class_list'])

    framedir_to_label = {ann['frame_dir']: ann['label'] for ann in data['annotations']}
    new_split = {}

    for split_name in split_names:
        split_dir = pjoin(out_path, split_name)
        for subset in ['train', 'val', 'test']:
            
            key = f"{split_name}_{subset}"
            txt_path = pjoin(split_dir, f"{subset}.txt")

            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    selected = set(x.strip() for x in f if x.strip())

                original = data['split'].get(key, [])
                new_split[key] = [x for x in original if framedir_to_label[x] not in class_list or x in selected]

    data['split'] = new_split
    data = filter_data_consistency(data)

    out_file = pjoin(out_path, "pyskl_data.pkl")
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


def create_unique_split_dir(base_dir, run_info):
    """Creates a new uniquely numbered subfolder (with 'S' prefix) and stores metadata."""
    os.makedirs(base_dir, exist_ok=True)
    existing = []
    for f in os.listdir(base_dir):
        if f.startswith('S') and f[1:].isdigit():
            existing.append(int(f[1:]))
    next_id = f"{(max(existing) + 1) if existing else 0:04d}"
    run_dir = pjoin(base_dir, f"S{next_id}")
    os.makedirs(run_dir)
    with open(pjoin(run_dir, "meta.json"), 'w') as f:
        json.dump(run_info, f, indent=2)

    return run_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NTU60', choices=['NTU60', 'NTU120', 'NTU-VIBE'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class_list', type=int, nargs='+', default=None, help='List of classes to include in the few-shot split. If not provided, all classes (except black-listed ones) will be used.')
    parser.add_argument('--shots', type=int, default=10, help='Number of shots per class for the few-shot split.')
    parser.add_argument('--with-stats', action='store_true', help='Whether to compute and save mean/std for the sampled split. Default is False.')
    parser.add_argument('--outliers', type=str, default='outliers.txt', help='Path to outliers file.')
    parser.add_argument('--exclude_outliers', type=bool, default=True, help='Whether to exclude outliers listed in the outliers file. Default is True.')
    args = parser.parse_args()
    random.seed(args.seed)

    # Hardcoded paths / data
    base_dataset_path = pjoin("data", args.dataset)

    annotations_path = SimpleNamespace(
        joint_vecs = pjoin(base_dataset_path, 'new_joint_vecs'),
        joints = pjoin(base_dataset_path, 'new_joints')
    )
    default_splits_base_path = pjoin(base_dataset_path, 'splits', 'default')
    fewshot_base_path = pjoin(base_dataset_path, 'splits', 'fewshot')
    split_sets = ['train', 'val', 'test']

    if args.class_list is None:
        # If no class list is provided, use all classes except ignored ones
        args.class_list = [i for i in range(DATA_NUM_CLASSES[args.dataset]) if i not in DATA_IGNORE_CLASSES[args.dataset]]
        print(f"\nNo class list provided, split will include {len(args.class_list)} classes: {args.class_list}\n(excluding): {DATA_IGNORE_CLASSES[args.dataset]}")

    assert not (set(args.class_list) & set(DATA_IGNORE_CLASSES[args.dataset])), "Class list contains ignored classes. Please remove them from the class list."
    args.class_list = sorted(set(args.class_list))        

    # Generate few-shot split
    print(f"\nGenerating few-shot split for {args.dataset} with {args.shots} shots per class.")
    run_metadata = {
        "class_list": args.class_list,
        "seed": args.seed,
        "shots": args.shots,
    }
    fewshot_splits_path = create_unique_split_dir(fewshot_base_path, run_metadata)
    os.makedirs(fewshot_splits_path, exist_ok=True)

    # Gather listed outliers
    if args.exclude_outliers:
        with open(pjoin(base_dataset_path, args.outliers), 'r') as f:
            outliers = set(line.strip() for line in f if line.strip())
    else:
        outliers = set()

    # 1. Parse split flies
    split_files, split_names = parse_split_files(default_splits_base_path, split_sets)
    assert split_files, "No default splits found."

    # 2. Sample from split files and save (lists + stats)
    print(f"Parsing split files...")
    for split_file in split_files:
        N = int(args.shots)
        split_name = os.path.basename(os.path.dirname(split_file))
        split_output_path = pjoin(fewshot_splits_path, split_name)
        os.makedirs(split_output_path, exist_ok=True)
        print(f"\n\t[{split_name}]")
        process_split_file(split_file, args.class_list, N, annotations_path, split_output_path, outliers, with_stats=args.with_stats)

    print(f"Done! generated few-shot split at {fewshot_splits_path}.")