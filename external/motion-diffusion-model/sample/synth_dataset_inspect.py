import os
import numpy as np
import torch
import shutil
from moviepy.editor import clips_array
from einops import rearrange
import random

import json
from os.path import join as pjoin
from utils import dist_util
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from data_loaders.get_data import get_single_stream_dataloader
from data_loaders.unified.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from utils.dataset_util.ntu_util import get_ntu_blacklist
from argparse import Namespace
import data_loaders.unified.utils.paramUtil as paramUtil
from data_loaders.unified.scripts.motion_process import recover_from_ric

from sample.generate import save_multiple_samples, construct_template_variables

def main(args=None):
    if args is None:
        args = generate_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    # Define which stream to use and in which mode
    s_stream = args.sampling_stream    
    sampling_stream_args = getattr(args, 'target') # FIXME hardcoded
    assert sampling_stream_args.dataset in ['ntu60', 'ntu120', 'ntu-vibe'], 'Only NTU60 and NTU120 are supported'

    device = dist_util.dev()

    SYNTH_DATA_ROOT = pjoin(
        '.', 'dataset', sampling_stream_args.dataset.upper(),
        'splits', 'synth', sampling_stream_args.fewshot_id, sampling_stream_args.task_split
    )
    assert os.path.exists(SYNTH_DATA_ROOT), f"Synthetic data root not found: {SYNTH_DATA_ROOT}"
    FPS = 20

    synth_data_candidates = []
    for d in os.listdir(SYNTH_DATA_ROOT):
        with open(pjoin(SYNTH_DATA_ROOT, d, 'info.json'), 'r') as f:
            synth_info = json.load(f) 
            if synth_info['model_path'] == args.model_path:
                synth_data_candidates.append(d)
                
    assert len(synth_data_candidates) > 0, f"No synthetic data found for model {args.model_path}"
    assert len(synth_data_candidates) == 1, f"Multiple synthetic data found for model {args.model_path}"

    # pick the first one by default
    SYNTH_IDX = 0 # hardcoded
    synth_data = synth_data_candidates[SYNTH_IDX]
    OUT_PATH = pjoin(SYNTH_DATA_ROOT, synth_data, 'viz')
    os.makedirs(OUT_PATH, exist_ok=True)
    print(">>> Synth data from: ", pjoin(SYNTH_DATA_ROOT, synth_data))

    sampling_stream_args.cond_mode = 'action' # force action condition on sampling
    data = get_single_stream_dataloader(
        data_stream_args=sampling_stream_args, 
        batch_size=64, # hardcoded
        split='train', hml_mode='train', 
        synth_data_folder = synth_data,
        device=device,
    )
    data.dataset.set_transforms([]) # nullify motion transoforms

    all_motions, all_lengths, all_text, all_ids = [], [], [], []
    print(f"\n===== GATHERING | From [{s_stream}] stream =====")
    for motion, cond in data:
        motion = rearrange(motion, 'b c j t -> b t (j c)')
        motion = recover_from_ric(motion, 22)
        for idx in range(motion.shape[0]):
            all_motions.append(motion[idx].cpu().numpy())
            all_lengths.append(int(cond['y']['lengths'][idx].cpu().numpy()))
            all_text.append(data.dataset.m_dataset.get_class_name(
                int(cond['y']['action'][idx].cpu().numpy()), is_compact_id=True)
            )
            if 'db_key' in cond['y'] and cond['y']['db_key'][0] is not None:
                 all_ids.append(cond['y']['db_key'][idx])
            else:
                 all_ids.append('N/A')

    all_motions = np.stack(all_motions)
    all_lengths = np.array(all_lengths)
    all_text = np.array(all_text)
    # Recalculate max_length later based on selected samples or keep global? 
    # Global is safer for consistent visualization.
    max_length = max(all_lengths) 
    
    print(f"saving visualizations to [{OUT_PATH}]...")
    skeleton = paramUtil.t2m_kinematic_chain
    animations = np.empty(shape=(args.num_rows, args.num_cols), dtype=object)
    _, _, _, sample_file_template, _, all_file_template = construct_template_variables(False)

# -------------------------------------------------------------------------
    # ### NEW LOGIC START ###
    
    # 1. Group indices by class
    class_to_indices = {}
    for i, text in enumerate(all_text):
        if text not in class_to_indices:
            class_to_indices[text] = []
        class_to_indices[text].append(i)

    # 2. Get sorted unique classes
    # We sort them to ensure the order is deterministic (always same order based on name)
    available_classes = sorted(list(class_to_indices.keys()))
    
    num_unique = len(available_classes)
    print(f"Found {num_unique} unique classes. generating {args.num_rows} rows cyclically.")

    # 3. Iterate rows (Cyclic Class Selection)
    for row in range(args.num_rows):
        # Use modulo to cycle through classes: 0, 1, 2, 0, 1, 2...
        cls_name = available_classes[row % num_unique]
        cls_indices = class_to_indices[cls_name]
        
        # 4. Select `num_cols` random samples for this class
        # If class has fewer samples than num_cols, sample with replacement
        replace_samples = len(cls_indices) < args.num_cols
        selected_indices = np.random.choice(cls_indices, args.num_cols, replace=replace_samples)

        # 5. Iterate columns (Samples)
        for column in range(args.num_cols):
            data_idx = selected_indices[column]
            
            # Retrieve the specific sample
            caption = all_text[data_idx]
            length = all_lengths[data_idx]
            motion = all_motions[data_idx][:max_length]
            current_id = all_ids[data_idx]

            # Padding logic
            if motion.shape[0] > length:
                motion[length:-1] = motion[length-1] 

            title = caption
            title = caption + ("" if current_id == 'N/A' else f" | ID: {current_id}")

            save_file = sample_file_template.format(row, column)
            animation_save_path = os.path.join(OUT_PATH, save_file)
            
            # Generate plot
            animations[row, column] = plot_3d_motion(
                animation_save_path, 
                skeleton, motion, 
                dataset=sampling_stream_args.dataset, 
                title=title, 
                fps=FPS, gt_frames=[], m_len=length
            )
            
    # ### NEW LOGIC END ###
    # -------------------------------------------------------------------------

    save_multiple_samples(OUT_PATH, {'all': all_file_template}, animations, FPS, max_length)

if __name__ == "__main__":
    main()