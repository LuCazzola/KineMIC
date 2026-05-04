import numpy as np
import torch
import os
import json
import argparse

from os.path import join as pjoin
from types import SimpleNamespace
from tqdm import tqdm

from scripts.skel_adaptation import resample_motion
from utils.humanml3d import process_text, motion_2_hml_vec, recover_from_ric, cal_mean_variance
from utils.constants.data import DATA_IGNORE_CLASSES, NTU_METADATA, DATASET_FPS
from utils.constants.skel import SKEL_INFO, FEET_THRE, FLOOR_THRE

# Functions to get info from NTU filenames
get_subject = lambda x: int(x.split('S')[1][:3])
get_setup = lambda x: int(x.split('C')[1][:3])
get_label = lambda x: int(x.split('A')[1][:3])

def skel_preproc(samples, motions, out_path):
    """
    Apply data pre-processing on skeleton data.
    """
    for idx, x in enumerate(tqdm(motions)):

        x = resample_motion(x, original_fps=DATASET_FPS[DATASET], target_fps=DATASET_FPS['HML3D'])        
        new_joint_vecs = motion_2_hml_vec(x, floor_thre=FLOOR_THRE, feet_thre=FEET_THRE)  # (T_new-1, 263)
        new_joints = recover_from_ric(torch.from_numpy(new_joint_vecs).unsqueeze(0).float(), SKEL_INFO["smpl"].joints_num).squeeze().numpy() # (T_new-1, 22, 3)
        new_joints = new_joints.reshape(new_joints.shape[0], -1) # (T_new-1, 22*3)
        # Store
        assert not np.isnan(new_joint_vecs).any() and not np.isnan(new_joints).any(), f"NaN values found in joint vectors for {samples[idx]}."
        np.save(pjoin(out_path.joint_vecs, f"{samples[idx]}.npy"), new_joint_vecs)
        np.save(pjoin(out_path.joints, f"{samples[idx]}.npy"), new_joints)


def format_default_splits(samples, out_path):
    """
    Transcribes NTU-VIBE splits to MDM format (.txt)
    """
    # IDs of the subjects in the validation split for xsub task (accordingly to the paper)
    ntu_xsub_train_subj = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
                        38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
                        80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
    
    get_train_split = {
        'xsub': lambda x: 'train' if get_subject(x) in ntu_xsub_train_subj else 'val',
        'xset': lambda x: 'train' if get_setup(x) % 2 == 0 else 'val' 
    }
    train_split = [get_train_split[t](name) for t in ['xsub', 'xset'] for name in samples]

    for t in ['xsub', 'xset']:
        for s in ['train', 'val']:
            split_dir = pjoin(out_path, t)
            os.makedirs(split_dir, exist_ok=True)
            # write down samples
            out_file_path = pjoin(split_dir, f"{s}.txt")
            with open(out_file_path, 'w') as f:
                for i, name in enumerate(samples):
                    if train_split[i] == s:
                        f.write(f"{name}\n")
            # write down labels
            out_label_path = pjoin(split_dir, f"{s}_y.txt")
            with open(out_label_path, 'w') as f:
                for i, name in enumerate(samples):
                    if train_split[i] == s:
                        f.write(f"{get_label(name)-1}\n") # NOTE: -1 for 0-indexed labels            

def compute_statistics(out_path, default_splits_path):
    '''
    Compute statistics for default splits
    '''
    for split in tqdm(os.listdir(default_splits_path), desc="Computing statistics for splits"):
        split_dir = pjoin(default_splits_path, split)
        for attr in ['joint_vecs', 'joints']:
            datapath = getattr(out_path, attr)
            with open(pjoin(split_dir, 'train.txt'), 'r') as f:
                fnames = [line.strip() for line in f.readlines()]
            fnames = [pjoin(datapath, fn + ".npy") for fn in fnames]
            motion = np.concatenate([np.load(fn) for fn in fnames], axis=0) # (N, T_new-1, 22*3)
            # Compute mean and std
            if attr == 'joint_vecs':
                # For hml vec. representation
                mean, std = cal_mean_variance(motion, SKEL_INFO["smpl"].joints_num)
            else :
                # for general representation (e.g., xyz)
                mean = np.mean(motion, axis=0)
                std = np.std(motion, axis=0)
            np.save(pjoin(split_dir, f"Mean_{attr}.npy"), mean)
            np.save(pjoin(split_dir, f"Std_{attr}.npy"), std)

def format_texts(samples, out_path, ntu_meta_info):
    """
    Given a action_captions file, transcribes data into .txt files using HumanML3D POS tagging logic.
    """
    with open(ntu_meta_info, 'r', encoding='utf-8') as f:
        ntu_info = json.load(f)

    for sample in tqdm(samples):
        entry = ntu_info['actions'][get_label(sample)-1] # NOTE: -1 for 0-indexed labels
        formatted_lines = []
        for cap in entry['captions']:
            cap = cap.lower() # Lowercase the caption
            word_list, pos_list = process_text(cap)
            pos_string = ' '.join(f'{word_list[i]}/{pos_list[i]}' for i in range(len(word_list)))
            formatted_lines.append(f"{cap}#{pos_string}#0.0#0.0")

        txt_path = pjoin(out_path, f"{sample}.txt")
        with open(txt_path, 'w', encoding='utf-8') as out_f:
            out_f.write("\n".join(formatted_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="setup options")
    parser.add_argument("--in_dataset", type=str, default='NTU-VIBE-RAW', choices=["NTU-VIBE-RAW"], help="SMPL skeletons extracted through VIBE")
    parser.add_argument("--out_dataset", type=str, default='NTU-VIBE', choices=["NTU-VIBE"], help="Output dataset name")
    parser.add_argument("--skip_skel_preproc", action='store_true', help="Skip skeleton pre-processing step")
    parser.add_argument("--label_filter", type=int, nargs='+', default=[99, 102, 104], help="Labels to consider")
    args = parser.parse_args()

    global DATASET, WHITELIST, BLACKLIST
    DATASET = args.out_dataset
    WHITELIST = set(range(1, 121) if len(args.label_filter) == 0 else args.label_filter) - set(DATA_IGNORE_CLASSES[DATASET])

    # 1.
    print(f"\nLoading {DATASET} dataset...")
    base_in_dataset_path = pjoin("data", args.in_dataset)
    base_out_dataset_path = pjoin("data", args.out_dataset)
    samples, motions = [], []
    for subject in tqdm(os.listdir(base_in_dataset_path)):
        for name in os.listdir(pjoin(base_in_dataset_path, subject)):
            if get_label(name) in WHITELIST:
                samples.append(name.replace('.npy', ''))
                motions.append(np.load(pjoin(base_in_dataset_path, subject, name)))
    print(f"> {len(samples)} samples loaded from {base_in_dataset_path}")

    # 2.
    print("\nFormatting default splits...")
    out_defsplit_path = pjoin(base_out_dataset_path, "splits", "default")
    os.makedirs(out_defsplit_path, exist_ok=True)
    # .
    format_default_splits(samples, out_defsplit_path)
    print(f"Default splits formatted and stored at {out_defsplit_path}")

    # Prepare output folders for annotations
    out_annotations_path = SimpleNamespace(
        joints = pjoin(base_out_dataset_path, "new_joints"),
        joint_vecs = pjoin(base_out_dataset_path, "new_joint_vecs")
    )
    os.makedirs(out_annotations_path.joints, exist_ok=True)
    os.makedirs(out_annotations_path.joint_vecs, exist_ok=True)

    if not args.skip_skel_preproc:
        # 4.
        print(f"\nApplying forward mapping on {DATASET} dataset...")
        skel_preproc(samples, motions, out_annotations_path)
        print(f"Forward mapping applied to {DATASET}")
        print(f"{out_annotations_path.joint_vecs} : joint vector representations (hml_vec format)")
        print(f"{out_annotations_path.joints} : joint position vectors (xyz)")

        # 5.    
        print(f"\nComputing statistics...")
        compute_statistics(out_annotations_path, out_defsplit_path)
        print(f"Statistics computed and stored at {out_defsplit_path} for both joint_vecs and joints.")

    # 6.
    print(f"\nFormatting texts...")
    ntu_meta_info = pjoin('.', 'data', NTU_METADATA)
    assert os.path.exists(ntu_meta_info), f"Input data {ntu_meta_info} not found."
    out_texts_path = pjoin(base_out_dataset_path, "texts")
    os.makedirs(out_texts_path, exist_ok=True)
    # .
    format_texts(samples, out_texts_path, ntu_meta_info)
    print(f"Annotations formatted and stored at {out_texts_path}")
    
    # Done
    print(f"\nDone! dataset {DATASET} stored at {base_out_dataset_path} .")