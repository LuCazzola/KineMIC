"""
Given a trained model and the dataset it was trained on, sample a synthetic to reproduce the full training set

Essentially, first it's checked what data the model used for training, which tipically is a fraction of the full dataset.
The data the model was trained on is blacklisted and the remaining data is used to sample the synthetic dataset.

Such approach makes sure the synthetic dataset will have the very same amount of samples per class as the real one
as well as the same data distribution (in terms of length, action types, etc.)
"""

import os
import numpy as np
import torch
import shutil
import json

from einops import rearrange
from tqdm import tqdm
from os.path import join as pjoin

from utils import dist_util
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from model.wrappers import wrap_w_classifier_free_sampling
from data_loaders.get_data import get_single_stream_dataloader
from utils.dataset_util.ntu_util import get_ntu_blacklist

def main(args=None):
    if args is None:
        args = generate_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    print("Model Type : ", args.model_type)
    
    s_stream = args.sampling_stream # Define which stream to use and in which mode
    assert s_stream == 'target', 'Build synth. data supported only for target dataset, as it is the one we want to expand'

    active_streams = [s_stream] if args.model_type in ['MDM'] else args.stream_names
    sampling_stream_args = getattr(args, s_stream)
    unconstrained = sampling_stream_args.unconstrained

    # To keep track of run info
    info = {
        'model_path': args.model_path,
        'dataset': sampling_stream_args.dataset,
        'data_rep': sampling_stream_args.data_rep,
        'task_split': sampling_stream_args.task_split,
        'fewshot_id': sampling_stream_args.fewshot_id,
    }

    # a blacklist consists in the names of samples not belonging to the same
    # class distribution as the few-shot training set
    blacklist, whitelise = get_ntu_blacklist(
        dataset=sampling_stream_args.dataset,
        fewshot_id=sampling_stream_args.fewshot_id,
        task_split=sampling_stream_args.task_split,
    )
    # This forces to load the full dataset, by accessing the "default" xsub split instead the few-shot split from training
    args.target.fewshot_id = None
    # NOTE: the combination of using the blacklist and setting fewshot_id=None results in a NTU Dataset object
    # which loads all samples from the full dataset except the ones in the blacklist (all real data for the fewshot classes)

    print("\n=== GOAL Data ==")
    data = {}
    for stream in active_streams :
        print(f"\n=== {stream.upper()} Data ===")
        data[stream] = get_single_stream_dataloader(
            data_stream_args=getattr(args, stream),
            batch_size=args.batch_size,
            split='train', hml_mode='train',
            device=dist_util.dev(),
            blacklist = blacklist['train'] if stream == s_stream else set(),
        )
    print("> Num samples: ", len(data[s_stream].dataset))

    print("\n===== Creating model and diffusion =====") 
    model, diffusion = create_model_and_diffusion(args, data, active_streams)
    sample_fn = diffusion.p_sample_loop
    data = data[s_stream] # simplify data access (when using 2 streams the target one is actually useless for sampling)

    print(f"\nCheckpoints from [{args.model_path}]...")
    model = load_saved_model(model, args.model_path, active_streams)
    sampling_model = wrap_w_classifier_free_sampling(model) if args.guidance_param != 1 else model
    # NOTE: model == sampling_model for base MDM, for KineMIC instead it references the target stream only
    model.to(dist_util.dev())
    model.eval()

    # Output storage
    OUT_PATH = format_out_path(args, info)
    JOINTS_DIR = 'new_joint_vecs' if getattr(sampling_stream_args, 'data_rep', 'hml_vec') == 'hml_vec' else 'new_joints'
    os.makedirs(os.path.join(OUT_PATH, JOINTS_DIR))
    synth_filename = lambda idx: 'synth_{:06d}'.format(idx)

    # Begin sampling
    all_text, all_actions = [], []
    sample_counter = 0
    print(f"\n===== SAMPLING | From [{s_stream}] stream, Mode [{args.sampling_mode}] =====")
    for motion, cond in tqdm(data):        
        
        # NOTE: no need to move motion to device, as we only need cond infor for sampling
        cond['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
        if not unconstrained and args.guidance_param != 1: # apply guidance for CFG sampling
            cond['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        # Begin sampling
        sample = sample_fn(
            sampling_model,
            motion.shape,
            clip_denoised=False,
            model_kwargs=cond,
            skip_timesteps=0,  # i.e. sampling from pure noise
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        sample = rearrange(sample.cpu(), 'batch feats joints time -> batch time (joints feats)').float()
        sample = data.dataset.m_dataset.de_normalize_motion(sample) # store in original space
        
        # Collect conditioning info
        curr_texts, curr_actions = [], []
        if 'text' == data.dataset.cond_mode:
            curr_texts = cond['y']['text']
        elif 'action' == data.dataset.cond_mode:
            curr_texts = cond['y']['action_text']
            curr_actions = [ # NOTE: Store the original action ids, not the compacted ones
                data.dataset.m_dataset.get_compact_class_id(a, reverse=True)
                for a in cond['y']['action'].squeeze().cpu().tolist()
            ]
        elif 'mixed' == data.dataset.cond_mode:
            curr_texts = cond['y']['text']
            curr_actions = [
                data.dataset.m_dataset.get_compact_class_id(a, reverse=True)
                for a in cond['y']['action'].squeeze().cpu().tolist()
            ]
        all_text.extend(curr_texts)
        all_actions.extend(curr_actions)
        
        # Collect and store samples
        curr_lenghs = cond['y']['lengths'].cpu().tolist()
        for i in range(sample.shape[0]):
            samplepath = os.path.join(OUT_PATH, JOINTS_DIR, synth_filename(sample_counter) + '.npy')
            np.save(samplepath, sample[i, :curr_lenghs[i], ...]) # store removing Pad.
            sample_counter += 1

    # Write back single files
    with open(pjoin(OUT_PATH, 'synth.txt'), 'w') as fw:
        fw.write('\n'.join([synth_filename(i) for i in range(sample_counter)]))
    with open(pjoin(OUT_PATH, 'synth_y.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_actions]))
    with open(pjoin(OUT_PATH, 'synth_text.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(pjoin(OUT_PATH, 'info.json'), 'w') as f:
        json.dump(info, f)
    
def format_out_path(args, info):
    
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model_', '').replace('.pt', '')
    out_path = os.path.join(
        '.', 'dataset', info['dataset'].upper(), 'splits', 'synth',
        info['fewshot_id'], info['task_split'], '{}_{}_seed{}'.format(name, niter, args.seed)
    )
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    return out_path

if __name__ == "__main__":
    main()
