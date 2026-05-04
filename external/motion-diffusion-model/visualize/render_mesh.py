import argparse
import os
from visualize import vis_utils
import shutil
from tqdm import tqdm

from os.path import join as pjoin

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_root", type=str, default=None, help='Root directory for .npy files')
    parser.add_argument("--npy_files", type=str, nargs='*', help='List of .npy filenames to process')
    parser.add_argument("--cuda", type=bool, default=False, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    args = parser.parse_args()

    
    npy_files = [pjoin(args.npy_root, fname) for fname in args.npy_files]
    for npy_file in npy_files:
        if not npy_file.endswith('.npy'):
            print(f"Skipping non-npy file: {npy_file}")
            continue
        
        motion = np.load(npy_file)
        npy2obj = vis_utils.npy2obj(npy_file, device=args.device, cuda=args.cuda)
        
        results_dir = pjoin('visualize', 'media', os.path.basename(npy_file).replace('.npy', '_obj')) 
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)

        print(f'Processing file: {npy_file}')
        for frame_i in tqdm(range(npy2obj.num_frames)):
            npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)