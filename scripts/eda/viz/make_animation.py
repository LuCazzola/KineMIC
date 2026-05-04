import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from moviepy.editor import clips_array

from textwrap import wrap
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

from utils.constants.data import DATA_FILENAME

COLORS = {
    5 : {
        'blue' : ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"],
        'orange' : ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
    },
    7 : {
        'blue' : ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A", "#FF0000", "#FF0000"],
        'orange' : ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E", "#FF0000", "#FF0000"]
    }
}

def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')

    title_per_frame = type(title) == list
    if title_per_frame:
        assert len(title) == len(joints), 'Title length should match the number of frames'
        title = ['\n'.join(wrap(s, 20)) for s in title]
    else:
        title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        # fig.suptitle(title, fontsize=10)  # Using dynamic title instead
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset in ['humanml', 'ntu60']:
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    kin_tree_len = len(kinematic_tree)
    colors = COLORS[kin_tree_len]['orange']

    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        if kin_tree_len == 5:
            colors[0] = COLORS[5]['blue'][0]
            colors[1] = COLORS[5]['blue'][1]
    elif vis_mode == 'gt':
        if kin_tree_len == 5 :
            colors = COLORS[5]['blue']
        
    n_frames = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]  # memorize original x,z pelvis values

    # locate x,z pelvis values of ** each frame ** at zero
    data[..., 0] -= data[:, 0:1, 0] 
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        # sometimes index is equal to n_frames/fps due to floating point issues. in such case, we duplicate the last frame
        index = min(n_frames-1, int(index*fps))
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        
        # Dynamic title
        if title_per_frame:
            _title = title[index]
        else:
            _title = title
        _title += f' [{index}]'
        fig.suptitle(_title, fontsize=10)

        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        used_colors = COLORS[kin_tree_len]['blue'] if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)
        plt.axis('off')
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


        return mplfig_to_npimage(fig)

    ani = VideoClip(update)
    
    plt.close()
    return ani

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
        clips.write_videofile(all_sample_save_path, fps=fps, threads=4, logger=None)
    
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


if __name__ == "__main__":

    import os
    import pickle
    import argparse

    from pathlib import Path
    from types import SimpleNamespace
    from os.path import join as pjoin
    from tqdm import tqdm

    from utils.constants.skel import kinematic_chain, FLOOR_THRE, SKEL_INFO
    from scripts.skel_adaptation.skel_mapping import resample_motion
    from utils.humanml3d.process_motion import globalize_pos

    ROOT = Path('.').resolve()
    SCRIPT_DIR = Path(__file__).parent.relative_to(ROOT)
    OUT_PATH = pjoin(SCRIPT_DIR, 'media')
    SIDE_BY_SIDE_PATH = pjoin(OUT_PATH, 'side_by_side')
    RAW_KINECT_PATH = pjoin(OUT_PATH, 'raw_kinect')

    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(SIDE_BY_SIDE_PATH, exist_ok=True)
    os.makedirs(RAW_KINECT_PATH, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--kine_dataset', type=str, default='NTU120', choices=['NTU60', 'NTU120'], help='Dataset to visualize')
    parser.add_argument('--smpl_dataset', type=str, default='NTU-VIBE', choices=['NTU60', 'NTU120', 'NTU-VIBE'], help='SMPL dataset to visualize')
    parser.add_argument('--raw-kinect', action='store_true', help='Save raw Kinect data as well (sample take directly from the dataset and untouched)')
    parser.add_argument('--samples', type=str, default='samples.txt', help='File .txt with sample names to visualize')
    parser.add_argument('--first_n', type=int, default=-1, help='Returns first N samples of the given file. negative for all')
    args = parser.parse_args()

    samples_path = pjoin(SCRIPT_DIR, args.samples)
    assert os.path.exists(samples_path), "Samples file not found: {}".format(samples_path)
    with open(pjoin(samples_path), 'r') as f:
        ntu_samples = [line.strip() for line in f.readlines() if line.strip()]
    if args.first_n < 0:
        args.first_n = len(ntu_samples)
    ntu_samples = ntu_samples[:args.first_n]

    DATA = {
        "kinect" : SimpleNamespace(
            dataset=args.kine_dataset.lower(),
            data_root=pjoin('.', 'data', args.kine_dataset, DATA_FILENAME[args.kine_dataset]),
            kinematic_chain=kinematic_chain['kinect'],
            n_joints=25,
            fps=30,
            samples=[
                SimpleNamespace(name=name)
                for name in ntu_samples
            ],
            num_samples=len(ntu_samples),
        ),

        "smpl" : SimpleNamespace(
            dataset=args.smpl_dataset.lower(),
            data_root=pjoin('.', 'data', args.smpl_dataset, 'new_joints'),
            texts_root=pjoin('.', 'data', args.smpl_dataset, 'texts'),
            kinematic_chain=kinematic_chain['smpl'],
            n_joints=22,
            fps=20,
            samples=[
                SimpleNamespace(name=name)
                for name in ntu_samples
            ],
            num_samples=len(ntu_samples)
        )
    }

    SIDE_BY_SIDE_FPS = DATA["smpl"].fps

    all_motions, all_lengths, all_text = {}, {}, {}
    for k in DATA.keys():
        all_motions[k], all_lengths[k], all_text[k] = [], [], []

    ##
    ## KINECT DATA
    ##

    with open(DATA['kinect'].data_root, 'rb') as file:
        raw_data = pickle.load(file)['annotations']
    kinect_data_lookup = {ann['frame_dir']: ann for ann in raw_data}
    raw_kinect_motion = []
    missing_entries = []
    for sample in tqdm(DATA['kinect'].samples, desc="Parsing Kinect data"):
        if sample.name in kinect_data_lookup:
            # read
            ann = kinect_data_lookup[sample.name]
            sample.caption = str(ann['label'])
            joints = ann['keypoint'][0]
            raw_kinect_motion.append(joints) # store apart raw motion
            # preprocess for better side-by-side comparison
            joints = resample_motion(joints, original_fps=DATA['kinect'].fps, target_fps=SIDE_BY_SIDE_FPS)  # resample to FPS
            joints = globalize_pos(joints, SKEL_INFO['kinect'], FLOOR_THRE) # feet on ground, facing Z+, ...
            joints[:,:,0] *= -1
            # store back
            all_motions['kinect'].append(joints)
            all_lengths['kinect'].append(joints.shape[0])
            all_text['kinect'].append(sample.caption)
        else:
            print(f"Warning: {sample.name} not found in kinect data")
            missing_entries.append(sample.name)

    ##
    ## SMPL DATA
    ##

    for sample in tqdm(DATA['smpl'].samples, desc="Parsing SMPL data"):

        if sample.name in missing_entries:
            print(f"Skipping {sample.name} as it was missing in Kinect data")
            continue

        joints = pjoin(DATA['smpl'].data_root, sample.name) + '.npy'
        text = pjoin(DATA['smpl'].texts_root, sample.name) + '.txt'
        assert os.path.exists(joints), "not found {}".format(joints)
        assert os.path.exists(text), "not found {}".format(text)
        # read
        joints = np.load(joints, allow_pickle=True)
        with open(text, 'r') as f:
            text = f.read().strip()
        sample.caption = text.split('#')[0] if '#' in text else text
        # store back
        all_motions['smpl'].append(joints)
        all_lengths['smpl'].append(joints.shape[0])    
        all_text['smpl'].append(sample.caption)

    ##
    ## VISUALIZE (RAW)
    ##

    if args.raw_kinect:
        for i, sample in tqdm(enumerate(DATA['kinect'].samples), desc="Generating Raw kinect animations", total=len(DATA['kinect'].samples)):
            save_path = pjoin(RAW_KINECT_PATH, f"{sample.name}.mp4")
            video_clip = plot_3d_motion(save_path,
                DATA['kinect'].kinematic_chain,
                raw_kinect_motion[i],
                dataset=DATA['kinect'].dataset,
                title=f"{sample.name} (raw Kinect) - '{sample.caption}' -",
                fps=DATA['kinect'].fps,
                gt_frames=[]
            )
            video_clip.duration = raw_kinect_motion[i].shape[0]/DATA['kinect'].fps    
            video_clip.write_videofile(save_path, fps=DATA['kinect'].fps, threads=4, logger=None)

    ##
    ## VISUALIZE (side-by-side)
    ##
    
    print(f"\nBuilding side-by-side visualizations...")
    args = SimpleNamespace(
        num_samples=min([data.num_samples for data in DATA.values()]),
        num_repetitions=len(list(DATA.keys())),
    )


    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(False)
    max_vis_samples = 6
    num_vis_samples = min(args.num_samples, max_vis_samples)
    animations = np.empty(shape=(args.num_samples, args.num_repetitions), dtype=object)
    max_length = max(all_lengths['kinect'] + all_lengths['smpl'])

    for sample_i in tqdm(range(args.num_samples)):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            format = 'kinect' if rep_i == 0 else 'smpl'
            data = DATA[format]
            caption = all_text[format][sample_i]
            # Trim / freeze motion if needed
            length = all_lengths[format][sample_i]
            motion = all_motions[format][sample_i][:max_length]#.transpose(2, 0, 1)[:max_length]
            
            if motion.shape[0] > length:
                motion[length:-1] = motion[length-1]  # duplicate the last frame to end of motion, so all motions will be in equal length

            save_file = sample_file_template.format(sample_i, rep_i)
            animation_save_path = os.path.join(SIDE_BY_SIDE_PATH, save_file)
            title = f"{DATA[format].samples[sample_i].name} ({format}) - '{caption}' -"
            animations[sample_i, rep_i] = plot_3d_motion(animation_save_path, 
                                                        data.kinematic_chain, motion, dataset=data.dataset, title=title, 
                                                        fps=SIDE_BY_SIDE_FPS, gt_frames=[])
            rep_files.append(animation_save_path)

    save_multiple_samples(SIDE_BY_SIDE_PATH, {'all': all_file_template}, animations, SIDE_BY_SIDE_FPS, max_length)
    print(f'[Done] Results are at [{os.path.abspath(SIDE_BY_SIDE_PATH)}]\n')