import random
import torch
import os
import random
import numpy as np
import json 
import wandb 

from utils.fixseed import fixseed
from utils import dist_util
from os.path import join as pjoin
from tqdm import tqdm
import pickle
from pathlib import Path
from einops import rearrange

import shutil
from data_loaders.unified.scripts.motion_process import recover_from_ric
import subprocess
import sys

import glob

def get_max_val_accuracy_from_log(log_path):
    val_accuracies = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # We only care about validation steps that have the top1_acc metric
                    if data.get('mode') == 'val' and 'top1_acc' in data:
                        val_accuracies.append(float(data['top1_acc']))
                except json.JSONDecodeError:
                    # Ignore lines that are not valid JSON (like the first env_info line)
                    continue
    except FileNotFoundError:
        print(f"\nWarning: Log file not found at {log_path}")
        return 0.0
    
    if not val_accuracies:
        print("\nWarning: No validation top1 accuracy found in the log file.")
        return 0.0
        
    return max(val_accuracies)


class LauncherPySKL():
    '''
    Tool to format our data into PySKL compatible format
    And launch training withing PySKL toolbox
    '''
    def __init__(self, args, opt):
        self.args = args
        self.opt = opt
        self.num_joints = 22 # hardcoded for SMPL (no hands, as in HumanML3D)

        self.pyskl_data = {
            'annotations' : [],
            'split' : {f'{self.opt.task_split}_train': [], f'{self.opt.task_split}_val': []} 
        }

    def _get_data(self, split_name, split_root, limit=None):
        with open( pjoin('.', split_root, f'{split_name}.txt'), 'r') as f:
            samples = [f.strip() for f in f.readlines()]
        with open(pjoin('.', split_root, f'{split_name}_y.txt'), 'r') as f:
            y = [int(f.strip()) for f in f.readlines()]
        
        if limit is not None and limit > 0 and limit < len(samples):
            samples = samples[:limit]
            y = y[:limit]
        
        return samples, y

    def _update_pyskl_data(self, sample_names, y, samples_root, joints_dir, eval_split):
        for idx, name in enumerate(tqdm(sample_names)):
            motion = np.load(pjoin(samples_root, joints_dir, f'{name}.npy')) # (T, D)
            if motion.shape[-1] == 263 : # hml_vec format 
                motion = recover_from_ric(torch.from_numpy(motion), self.num_joints).numpy() # (T, J, 3)
            elif motion.shape[-1] == 66 : # smpl (22 joints * 3) format (no hands, as in HumanML3D)
                motion = rearrange(motion, 'f (j c) -> f j c', j=self.num_joints) 
            else:
                raise NotImplementedError(f'Unknown joint format with {motion.shape[-1]} dims')

            motion = np.expand_dims(motion, axis=0) # add unit dimension for "number of skeletons" (fixed to 1 in our case)
            self.pyskl_data['annotations'].append({
                'frame_dir': name,
                'label': y[idx],
                'keypoint': motion,
                'total_frames': motion.shape[1]
            })
        self.pyskl_data['split'][f'{self.opt.task_split}_{eval_split}'] += sample_names

    def _compact_class_ids(self) :
        """
        Compacts class IDs to be sequential starting from 0.
        """
        unique_labels = sorted(list(set(ann['label'] for ann in self.pyskl_data['annotations'])))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        for ann in self.pyskl_data['annotations']:
            ann['label'] = label_mapping[ann['label']]
        return label_mapping

    def train(self):
        """
        Launches the PySKL training script, and after it completes,
        finds the final .json log, logs metrics, and moves it to the log directory.
        """
        # --- 1. Define paths and create log directory ---
        launch_script_path = os.path.abspath(pjoin(self.args.pyskl, 'tools', 'dist_train.sh'))
        # Define the destination directory for the final log file
        log_dir = os.path.abspath(pjoin(self.args.pyskl, 'logs', f'{self.args.model_har}_{self.opt.dataset_name}_{self.opt.fewshot_id}'))
        os.makedirs(log_dir, exist_ok=True)
        
        configs = pjoin(
            '.', 'configs', self.args.model_har, 
            f'{self.args.model_har}_kinemic_{self.opt.dataset_name.lower()}_{self.opt.task_split}_smpl',
            f'{self.args.data_rep}.py'
        )

        command = [
            'bash', launch_script_path, configs, '1',
            '--validate', # toggle validation during training
            '--seed', str(self.args.seed),
        ]

        try:
            # Change to the pyskl directory
            original_cwd = os.getcwd()
            os.environ['PYTHONPATH'] = self.args.pyskl
            os.chdir(self.args.pyskl)
            # Define and remove the previous work directory to ensure a fresh run
            workdir = configs.replace('configs', 'work_dirs').replace('.py', '')
            if os.path.exists(workdir):
                shutil.rmtree(workdir)
            print(f"\nLaunching training with command: {' '.join(command)}")
            subprocess.run(command, check=True)

            # The training script creates a log file like `YYYYMMDD_HHMMSS.log.json`
            json_files = glob.glob(pjoin(workdir, '*.json'))
            if json_files:
                # Find the most recently created JSON file
                source_json_path = max(json_files, key=os.path.getctime)
                max_acc = get_max_val_accuracy_from_log(source_json_path)
                print(f"> Max Run (validation) top1-accuracy: {max_acc:.4f}")

                # Log to W&B
                if self.args.use_wandb:
                    wandb.log({"val_top1_acc": max_acc})
                    wandb.summary["val_top1_acc"] = max_acc

                # Store it as a run log separately in pyskl/logs/...
                dest_json_path = pjoin(log_dir, f'{self.opt.run_name}.json')
                # Move and rename the file
                shutil.move(source_json_path, dest_json_path)
                print(f"\nTraining complete. Metrics log saved to: {dest_json_path}")
            else:
                print("\nWarning: No .json log file was found in the work directory after training.")

        except subprocess.CalledProcessError as e:
            print(f"Bash script failed with exit code {e.returncode}. Training was not successful.")
            sys.exit(1)
        finally:
            # Always restore the original directory
            os.chdir(original_cwd) # Restore original path
            os.environ['PYTHONPATH'] = original_cwd

    def __call__(self) :     

        '''
        Gather data samples from the specified splits
        '''

        val_samples, val_y = self._get_data('val', self.opt.real_split_root)
        real_train_samples, real_train_y = [], []
        
        if self.opt.requires_real:
            if self.args.incremental < 0:
                real_train_samples, real_train_y = self._get_data('train', self.opt.real_split_root)
            else:
                real_train_samples, real_train_y = self._get_data('incremental', self.opt.real_split_root, limit=self.args.incremental)
                if len(real_train_samples) < self.args.incremental:
                    # if the actual number of available samples is less than requested, use all available samples
                    print(f"Warning -> you requested {self.args.incremental} real samples, but only {len(real_train_samples)} are available, using all available samples...")
                    self.args.incremental = len(real_train_samples)

        synth_train_samples, synth_train_y = [], []      
        if self.opt.requires_synthetic:
            synth_train_samples, synth_train_y = self._get_data('synth', self.opt.synth_split_root)

        '''
        Fill the pyskl dataset
        '''

        print('Setting up [val] data...')
        self._update_pyskl_data(val_samples, val_y, self.opt.real_data_path, 'new_joints', 'val')
        if self.opt.requires_real:
            print(f'Setting up [real] training data ({len(real_train_samples)} samples)...')
            self._update_pyskl_data(real_train_samples, real_train_y, self.opt.real_data_path, 'new_joints', 'train')
        if self.opt.requires_synthetic:
            print(f'Setting up [synth] training data ({len(synth_train_samples)} samples)...')
            self._update_pyskl_data(synth_train_samples, synth_train_y, self.opt.synth_data_path, 'new_joint_vecs', 'train')

        '''
        pre-processing and launch training
        '''

        # Compact class IDs to be sequential starting from 0
        _ = self._compact_class_ids()

        # Store a temporary dataset file for easy loading in PySKL
        tmp_dataset = pjoin(self.args.pyskl, 'tmp_dataset.pkl')
        try:
            with open(tmp_dataset, 'wb') as f:
                pickle.dump(self.pyskl_data, f)
            print(f"\nTemporary dataset save to: {tmp_dataset}")
        except Exception as e:
            print(f"\nAn error occurred while saving : {e}")

        # Launch training within PySKL through OS instructions
        self.train()

        # Delete the temporary dataset file
        if not self.args.keep_data:
            try:
                os.remove(tmp_dataset)
                print(f"\nTemporary dataset {tmp_dataset} deleted.")
            except Exception as e:
                print(f"\nAn error occurred while deleting temporary dataset: {e}")


# incremental [9 18 60 90 120 150 225 300 375 450 600 750 900 1200]

if __name__ == "__main__":
    
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./datasets/<DATASET>/splits/<SPLIT>/...', help='Directory pointing to the data')
    parser.add_argument('--mode', type=str, default='mixed', choices=['gt', 'real', 'synth', 'mixed'], help='Training mode to use')
    parser.add_argument('--incremental', nargs='*', type=int, default=[-1], help='Amount of real data to use, set negative to use all available data')
    parser.add_argument('--data_rep', type=str, default='j', choices=['b', 'bm', 'j', 'jm'], help='Data representation format')
    parser.add_argument('--model_har', type=str, default='stgcn', choices=['stgcn'], help='Model architecture to use')
    parser.add_argument('--pyskl', type=str, default='../pyskl', help='Path to local pyskl')
    parser.add_argument('--seed', nargs='*', type=int, default=[1,2,3,4,5], help='Seeds')     
    parser.add_argument('--keep_data', action='store_true', help='Whether to keep the temporary dataset file after training')
    
    # W&B CHANGE: Add arguments for wandb
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='downstream_har_eval', help='W&B project name')
    
    args = parser.parse_args()
    
    opt = {}
    opt['requires_synthetic'] = (args.mode in ['synth', 'mixed'])
    opt['requires_real'] = (args.mode in ['real', 'mixed', 'gt'])
    if opt['requires_synthetic']:
        # in this cases, data_dir should point to the synthetic data folder
        datapath = Path(args.data_dir)
        assert 'synth' in datapath.parts, "For modes involving synthetic data, --data_dir should point to the synthetic data folder"
        # ex: ./dataset/<dataset_name>/splits/synth/<fewshot_id>/<task_split>/<synth_dataset>
        opt['synth_data_path'] = datapath
        opt['model_gen'] = datapath.parts[-1] # (the model which generated the synthetic data)
        opt['task_split'] = datapath.parts[-2]
        opt['fewshot_id'] = datapath.parts[-3]
        opt['dataset_name'] = datapath.parts[-6]
        opt['synth_split_root'] = datapath
        opt['real_data_path'] = datapath.parents[4]
        opt['real_split_root'] = opt['real_data_path'] / 'splits' / 'fewshot' / opt['fewshot_id'] / opt['task_split']
    elif args.mode in ['real', 'gt']:
        # in this cases, data_dir should point to the real data folder
        datapath = Path(args.data_dir)
        assert 'fewshot' in datapath.parts, "For modes involving only real data, --data_dir should point to the real data folder"
        # ex: ./dataset/<dataset_name>/splits/fewshot/<fewshot_id>/<task_split>
        opt['model_gen'] = 'real_data' # Give a name for real data runs
        opt['task_split'] = datapath.parts[-1]
        opt['fewshot_id'] = datapath.parts[-2]
        opt['dataset_name'] = datapath.parts[-5]
        opt['real_data_path'] = datapath.parents[3]  
        opt['real_split_root'] = datapath
    
    # Ground truth data split (info aboud the whole 'task_split')
    opt['gt_split_root'] = opt['real_data_path'] / 'splits' / 'default' / opt['task_split']

    opt['device'] = dist_util.dev()
    dist_util.setup_dist(opt['device'])
    
    # Launch training for each seed
    all_seeds = args.seed
    all_increments = args.incremental
    for curr_seed in all_seeds:
        args.seed = curr_seed
        fixseed(args.seed)

        for current_inc in all_increments:
            args.incremental = current_inc

            run_name = f"{opt['model_gen']}-{args.mode}-seed{args.seed}"
            run_name += (f'-inc:{args.incremental}' if args.incremental > 0. else '')
            opt['run_name'] = run_name

            if args.use_wandb:
                # Create a unique, informative name for the run
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config={**vars(args), **opt} # Log all args and options
                )
            try:
                launcher = LauncherPySKL(args, argparse.Namespace(**opt))
                launcher() # <- launch
            finally:
                if args.use_wandb:
                    wandb.finish()