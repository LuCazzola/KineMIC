# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import numpy as np
import torch
import shutil
import time
from moviepy.editor import clips_array
from einops import rearrange

import data_loaders.unified.utils.paramUtil as paramUtil
from utils import dist_util
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from model.wrappers import wrap_w_classifier_free_sampling
from data_loaders.get_data import get_single_stream_dataloader
from data_loaders.unified.scripts.motion_process import recover_from_ric, get_target_location, sample_goal
from data_loaders.unified.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate


def main(args=None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    out_path = args.output_dir

    print("Model Type : ", args.model_type)
    # Define which stream to use and in which mode
    s_stream = args.sampling_stream
    active_streams = [s_stream] if args.model_type in ['MDM'] else args.stream_names
    dataset = getattr(args, s_stream).dataset
    unconstrained = args.unconstrained_sampling

    N_JOINTS, MAX_FRAMES, FPS = 22, 196, 20  # FIXME: hardcoded for now
    n_frames = min(MAX_FRAMES, int(args.motion_length*FPS))
    
    texts, actions = None, None
    num_texts, num_actions = 0, 0
    # T2M
    if args.text_prompt != '':
        texts = [args.text_prompt]
        num_texts = 1
    # A2M
    if args.action_id:
        actions = [a-1 for a in args.action_id]
        num_actions = len(actions)

    # Determine the number of samples to generate
    if num_texts > 0 and num_actions == 0:
        args.num_samples = num_texts
    elif num_texts == 0 and num_actions > 0:
        args.num_samples = num_actions
    elif num_texts > 0 and num_actions > 0:
        # Create all combinations of text and action
        new_actions, new_texts = [], []
        for a in actions:
            for t in texts:
                new_actions.append(a)
                new_texts.append(t)
        actions, texts = new_actions, new_texts
        args.num_samples = len(actions)

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    total_num_samples = args.num_samples * args.num_repetitions

    data = load_dataset(args, active_streams, n_frames, mode='eval', split='val')
    print("\n===== Creating model and diffusion =====")
    model, diffusion = create_model_and_diffusion(args, data, active_streams, force_single_stream=None)
    sample_fn = diffusion.p_sample_loop

    print(f"\nCheckpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, active_streams)
    if args.guidance_param != 1:
        sampling_model = wrap_w_classifier_free_sampling(model)  # wrapping model with the classifier-free sampler
    data = data[s_stream] # NOTE: overwite the data dict, as only the sampling dataloader matters

    out_path = format_out_path(args, out_path)
    model.to(dist_util.dev())
    model.eval()

    #
    # Configure Conditioning / Model Inputs
    #
    cond_mode = data.dataset.cond_mode
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name, args.action_id])
    max_length = -1
    model_kwargs = {}
    
    if is_using_data:
        # No external conditioning was provided to the script
        # in this case, sample from the first batch of the dataloader
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        # If some external conditioning is provided, use it to create the batch
        # (e.g. text prompts, action labels, motion length)
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        
        if not unconstrained:
            
            assert not cond_mode == 'mixed' or (texts and actions and len(texts) == len(actions)), \
                "For mixed generation, both text and action must be provided"
            
            if cond_mode in ['text', 'mixed']:
                # t2m
                assert any([args.input_text, args.text_prompt]), "No text provided"
                collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
            
            if cond_mode in ['action', 'mixed']:
                # a2m
                action_text = [data.dataset.m_dataset.get_class_name(a) for a in actions]
                action = [data.dataset.m_dataset.get_compact_class_id(a) for a in actions]

                collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                                arg, one_action, one_action_text in zip(collate_args, action, action_text)]

        _, model_kwargs = collate(collate_args)

    # To tensor, to device
    model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

    if not unconstrained:
        # add CFG scale to batch     
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        # pre-compute text embeddings
        if cond_mode in ['text', 'mixed']:
            model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])


    feats = 66 if data.dataset.data_rep == 'xyz' else 263 # FIXME: hardcoded
    n_frames = model_kwargs['y']['lengths'].max().item()
    motion_shape = (args.batch_size, feats, 1, n_frames)

    # NEW: Lists to store raw motion vectors and action labels
    all_motions, all_raw_vectors, all_lengths, all_text, all_actions = [], [], [], [], []
    
    print(f"\n===== SAMPLING | From [{s_stream}] stream, Mode [{args.sampling_mode}] =====")
    for rep_i in range(args.num_repetitions):
        print(f'### repetitions #{rep_i}')
        start_time = time.perf_counter()
        sample = sample_fn(
            sampling_model, # NOTE: needed to use the wrapped model for CFG sampling
            motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None, # start from pure noise
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        print(f"Execution time: {(time.perf_counter() - start_time):.6f} seconds")

        # NEW: Capture the raw motion (before rot2xyz)
        raw_sample = rearrange(sample.cpu(), 'batch feats joints time -> batch time (joints feats)').float()
        raw_sample = data.dataset.m_dataset.de_normalize_motion(raw_sample)
        all_raw_vectors.append(raw_sample.cpu().numpy())
        
        # NEW: Capture the action label for metadata file
        if not unconstrained and 'action' in model_kwargs['y']:
            action_value = model_kwargs['y']['action']
            action_tensor = action_value if torch.is_tensor(action_value) else torch.tensor(action_value)
            all_actions.append(action_tensor.cpu().numpy())
        else:
            # Placeholder for unconstrained/text-only generation
            all_actions.append(np.array([-1] * args.batch_size))


        # OLD CODE CONTINUES: Prepare for rot2xyz (for visualization)
        
        if data.dataset.data_rep == 'xyz':
            sample = rearrange(raw_sample, 'batch time (joints feats) -> batch time joints feats', feats=3)
        elif data.dataset.data_rep == 'hml_vec':
            sample = recover_from_ric(raw_sample, N_JOINTS) # get XYZ from hml_vec representation
        else:
            pass
            
        sample = rearrange(sample, 'batch time joints feats -> batch joints feats time')

        rot2xyz_pose_rep = 'xyz' if data.dataset.data_rep in ['xyz', 'hml_vec'] else data.dataset.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                get_rotations_back=False)

        stream_cond = data.dataset.cond_mode
        if unconstrained:
            all_text += ['unconstrained'] * args.batch_size
        elif stream_cond in ['text', 'action']:
            cond_key = 'text' if stream_cond == 'text' else 'action_text'
            all_text += [f'{t}' for t in model_kwargs['y'][cond_key]]
        elif stream_cond == 'mixed':
            all_text += [f'{a} | {t}' for a, t in zip(model_kwargs['y']['action_text'], model_kwargs['y']['text'])]
        
        all_motions.append(sample.cpu().numpy())
        _len = model_kwargs['y']['lengths'].cpu().numpy()
        all_lengths.append(_len)

        print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    
    # NEW: Concatenate and trim raw vectors and actions
    all_raw_vectors = np.concatenate(all_raw_vectors, axis=0)[:total_num_samples] # [bs, seqlen, D]
    all_actions = np.concatenate(all_actions, axis=0)[:total_num_samples]

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"\nsaving results file to [{npy_path}]")
    np.save(npy_path, {
        'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
        'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions
    })
    text_file_content = '\n'.join(all_text)
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write(text_file_content)
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    # NEW: Call the new function to save individual motions and metadata
    save_individual_motions(out_path, all_raw_vectors, all_actions)
    
    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.t2m_kinematic_chain

    _, _, _, sample_file_template, _, all_file_template = construct_template_variables(unconstrained)
    animations = np.empty(shape=(args.num_samples, args.num_repetitions), dtype=object)
    max_length = max(all_lengths)

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            
            # Trim / freeze motion if needed
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:max_length]
            if motion.shape[0] > length:
                motion[length:-1] = motion[length-1]  # duplicate the last frame to end of motion, so all motions will be in equal length

            save_file = sample_file_template.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            animations[sample_i, rep_i] = plot_3d_motion(animation_save_path,
                                                             skeleton, motion, dataset=dataset, title=caption,
                                                             fps=FPS, gt_frames=[], m_len=length)
            rep_files.append(animation_save_path)

    save_multiple_samples(out_path, {'all': all_file_template}, animations, FPS, max(list(all_lengths) + [n_frames]))

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]\n')

    return out_path

# NEW: Helper function to save individual motion files and metadata
def save_individual_motions(out_path, all_raw_vectors, all_actions):
    samples_dir = os.path.join(out_path, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    print(f"\nSaving individual motion files to [{samples_dir}]...")
    synth_filenames = []
    synth_labels = []
    for i, motion_data in enumerate(all_raw_vectors):
        # motion_data is [seqlen, D]
        label = all_actions[i]
        # Use the global index as the filename
        filename = str(i)
        motion_save_path = os.path.join(samples_dir, f'{filename}.npy')
        # Save raw motion vector (de-normalized, [T, D])
        np.save(motion_save_path, motion_data)
        synth_filenames.append(filename)
        synth_labels.append(str(label))
    # Save the split info files required by the HAR downstream script
    synth_txt_path = os.path.join(out_path, 'synth.txt')
    synth_y_txt_path = os.path.join(out_path, 'synth_y.txt')

    print(f"Saving synthetic split info files: [synth.txt] and [synth_y.txt]")
    
    with open(synth_txt_path, 'w') as fw:
        fw.write('\n'.join(synth_filenames))
    with open(synth_y_txt_path, 'w') as fw:
        fw.write('\n'.join(synth_labels))


def save_multiple_samples(out_path, file_templates,  animations, fps, max_frames, no_dir=False):
    
    num_samples_in_out_file = 3
    n_samples = animations.shape[0]
    
    for sample_i in range(0,n_samples,num_samples_in_out_file):
        last_sample_i = min(sample_i+num_samples_in_out_file, n_samples)
        all_sample_save_file = file_templates['all'].format(sample_i, last_sample_i-1)
        if no_dir and n_samples <= num_samples_in_out_file:
            all_sample_save_path = out_path
        else:
            all_sample_save_path = os.path.join(out_path, all_sample_save_file)
            print(f'saving {os.path.split(out_path)[1]}/{all_sample_save_file}')

        clips = clips_array(animations[sample_i:last_sample_i])
        clips.duration = max_frames/fps
        
        # import time
        # start = time.time()
        clips.write_videofile(all_sample_save_path, fps=fps, threads=4, logger=None)
        # print(f'duration = {time.time()-start}')
        
        for clip in clips.clips: 
            # close internal clips. Does nothing but better use in case one day it will do something
            clip.close()
        clips.close()  # important
    

def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
             sample_file_template, row_file_template, all_file_template

def format_out_path(args, out_path):
    if out_path == '':
        name = os.path.basename(os.path.dirname(args.model_path))
        niter = os.path.basename(args.model_path).replace('model_', '').replace('.pt', '')

        out_path = os.path.join(
            os.path.dirname(args.model_path),
            'samples_{}_{}_seed{}'.format(name, niter, args.seed)
        )
        
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    return out_path

def load_dataset(args, active_streams, n_frames, mode=None, split=None):
    data = {}
    for stream in active_streams :
        stream_args = getattr(args, stream)
        print(f"\n=== {stream.upper()} Data ===")
        print(f"creating [{stream}] data loader...")
        data[stream] = get_single_stream_dataloader(
            data_stream_args=stream_args,
            batch_size=args.batch_size,
            split=split, hml_mode=mode,
            device=dist_util.dev()
        )
        # NOTE: needed when using 'cond_only' mode
        data[stream].fixed_length = n_frames
        
    return data

def is_substr_in_list(substr, list_of_strs):
    return np.char.find(list_of_strs, substr) != -1  # [substr in string for string in list_of_strs]

if __name__ == "__main__":
    main()