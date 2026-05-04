import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
from scipy import stats
from utils.humanml3d.process_motion import recover_from_ric, cal_mean_variance, recover_velocities

import random
import logging
from types import SimpleNamespace

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from os.path import join as pjoin
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

def motion_stats_from_xyz(positions):
    def compute_derivatives(x):
        def diff_and_norm(data):
            d = np.diff(data, axis=0)
            return d, np.linalg.norm(d, axis=2)
        
        vel_vec, vel_mag = diff_and_norm(x)
        acc_vec, acc_mag = diff_and_norm(vel_vec)
        jerk_vec, jerk_mag = diff_and_norm(acc_vec)

        return (
            {'vec': vel_vec, 'mag': vel_mag},
            {'vec': acc_vec, 'mag': acc_mag},
            {'vec': jerk_vec, 'mag': jerk_mag}
        )

    if isinstance(positions, np.ndarray):
        return compute_derivatives(positions)

    elif isinstance(positions, list):
        all_vel, all_acc, all_jerk = [], [], []
        for x in positions:
            vel, acc, jerk = compute_derivatives(x)
            all_vel.append(vel)
            all_acc.append(acc)
            all_jerk.append(jerk)

        def stack_dicts(dicts):
            return {
                'vec': [d['vec'] for d in dicts],
                'mag': [d['mag'] for d in dicts]
            }

        return stack_dicts(all_vel), stack_dicts(all_acc), stack_dicts(all_jerk)

    else:
        raise ValueError("Input must be a numpy array or a list of numpy arrays")

class MotionDatasetAnalyzer:
    def __init__(self, ntu_dir, hml_dir, cache_dir, out_path, args, n_joints=22, seed=42):
        
        self.n_joints = 22 # assumes smpl joints
        self.num_feats = 3 * self.n_joints if args.data_rep == 'xyz' else 263
        
        self.data_root = SimpleNamespace(
            ntu=Path(ntu_dir),
            hml=Path(hml_dir)
        )
        
        self.filter_set = SimpleNamespace(
            ntu=pjoin(out_path, args.ntu_set) if args.ntu_set != '' else '',
            hml=pjoin(out_path, args.hml_set) if args.hml_set != '' else ''
        )

        self.out_dir = Path(pjoin(out_path, args.data_rep))
        self.outliers_dir = pjoin(self.out_dir, 'outliers')

        cache_dir = pjoin(self.out_dir, cache_dir)
        self.cache = SimpleNamespace(
            ntu=pjoin(cache_dir, f'ntu_{args.ntu_set.replace(".txt","")}.npy'),
            hml=pjoin(cache_dir, f'hml_{args.hml_set.replace(".txt","")}.npy')
        )

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.outliers_dir, exist_ok=True)        
        
        self.color = {'NTU': 'blue', 'HML': 'red'}
        self.data = SimpleNamespace(ntu=[], hml=[])
        self.filenames = SimpleNamespace(ntu=[], hml=[])
        self.stats = SimpleNamespace(ntu={}, hml={})
        self.args = args

        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(pjoin(self.out_dir, 'summary.log')),
                logging.StreamHandler()
            ]
        )

        np.random.seed(seed)
    
    def load_datasets(self):
        """Load all .npy files from both directories"""

        def read_files(data_path, filter_set, data_name, cache):
            
            if os.path.exists(cache) and self.args.use_cache:
                logging.info(f"Loading cached {data_name} data from {cache}")
                cache_data = np.load(cache, allow_pickle=True).item()
                return cache_data['motion'], cache_data['filename']
            
            filenames = []
            if os.path.exists(filter_set):
                with open(filter_set, 'r') as f:
                    filenames = [str(line.strip()+'.npy') for line in f.readlines() if line.strip()]
            else:
                filenames = [fn for fn in os.listdir(data_path) if fn.endswith('.npy')]
            
            all_data = []
            unexpected_shapes = set()
            for fn in tqdm(filenames, desc=f"Loading {data_name} "):
                data = np.load(pjoin(data_path, fn), allow_pickle=True)
                T = data.shape[0]
                if data.shape not in [(T, self.n_joints, 3), (T, self.n_joints*3), (T, 263)]:
                    unexpected_shapes.add(data.shape)
                    continue

                if data.shape == (T, self.n_joints*3) and self.args.data_rep == 'xyz':
                    data = data.reshape(T, self.n_joints, 3)  # Reshape to (T, J, 3) if needed
                all_data.append(data)
            filenames = [fn.replace('.npy', '') for fn in filenames] # clean extension

            np.save(cache, {'motion' : all_data, 'filename' : filenames}, allow_pickle=True)

            if len(unexpected_shapes) > 0:
                logging.info(f"Unexpected shapes in HML dataset: {unexpected_shapes}")

            return all_data, filenames

        self.data.ntu, self.filenames.ntu = read_files(self.data_root.ntu, self.filter_set.ntu, 'NTU', self.cache.ntu)
        self.data.hml, self.filenames.hml = read_files(self.data_root.hml, self.filter_set.hml, 'HML', self.cache.hml)

        # Done loading
        logging.info(f"Loaded {len(self.data.ntu)} NTU sequences and {len(self.data.hml)} HML sequences")    

    def analyze_velocity_outliers(self, motion_samples, percentile=95.0):
        """
        Flags outlier frames if their velocities exceed the specified percentile threshold.
        """
        all_velocities, *_ = motion_stats_from_xyz(motion_samples)
            
        # Compute percentile bounds
        vel_thresh = np.percentile(
            np.linalg.norm(
                np.concatenate(all_velocities['vec'], axis=0), # Flattened across frame dimension
                axis=1
            ), percentile
        )
        logging.info(f"Velocity threshold for outliers (percentile {percentile}%): {vel_thresh:.4f}")

        # Check each sample against bounds
        outliers = []
        num_outlier_frames, affected_samples = 0, 0
        for vel in all_velocities['vec']:
            vel_magnitude = np.linalg.norm(vel, axis=1)
            outlier_frames = np.where(vel_magnitude > vel_thresh)[0]
            outliers.append(outlier_frames)
            num_outlier_frames += len(outlier_frames)
            if len(outlier_frames) > 0:
                affected_samples += 1

        logging.info(num_outlier_frames, "outlier frames detected across all samples")
        total_frames = sum([vel.shape[0] for vel in all_velocities['vec']])
        logging.info(f"Outlier ratio: {(num_outlier_frames / total_frames):.5%} of all frames")            
        logging.info(f"{affected_samples} samples have outlier frames, being {(affected_samples / len(motion_samples)):.2%} of all samples")
        logging.info("---")

    
    def compute_basic_statistics(self):
        """Compute basic statistics for both datasets"""
        def compute_stats(data_list):
            stats_dict = {}
            
            ##
            ## LENGTH STATISTICS
            ##

            lengths = [seq.shape[0] for seq in data_list]
            stats_dict['lengths'] = {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'median': np.median(lengths),
                'q25': np.percentile(lengths, 25),
                'q75': np.percentile(lengths, 75)
            }
            
            ##
            ## FEATURES STATISTICS
            ##

            all_data = np.concatenate(data_list, axis=0)            
            
            if self.args.data_rep == 'hml_vec':
                mean, std = cal_mean_variance(all_data, self.n_joints)
            else:
                mean = np.mean(all_data, axis=0)
                std = np.std(all_data, axis=0)

            stats_dict['global'] = {
                'mean': mean,
                'std': std,
                'min': np.min(all_data, axis=0),
                'max': np.max(all_data, axis=0),
                'total_frames': all_data.shape[0]
            }
            
            return stats_dict
        
        self.stats.ntu = compute_stats(self.data.ntu)
        self.stats.hml = compute_stats(self.data.hml)
        
    def plot_sequence_length_distribution(self):
        """Plot sequence length distributions for both datasets"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        lengths = SimpleNamespace(
            ntu=[seq.shape[0] for seq in self.data.ntu],
            hml=[seq.shape[0] for seq in self.data.hml]
        )

        # Histograms
        axes[0].hist(lengths.ntu, bins=100, alpha=0.7, label='NTU', color=self.color['NTU'])
        axes[0].hist(lengths.hml, bins=100, alpha=0.7, label='HML', color=self.color['HML'])
        axes[0].set_xlabel('Sequence Length')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Sequence Length Distribution')
        axes[0].legend()
        
        # Box plots
        axes[1].boxplot([lengths.ntu, lengths.hml], labels=['NTU', 'HML'])
        axes[1].set_ylabel('Sequence Length')
        axes[1].set_title('Sequence Length Box Plot')
        
        # CDF comparison
        lengths.ntu = np.sort(lengths.ntu)
        lengths.hml = np.sort(lengths.hml)
        ntu_cdf = np.arange(1, len(lengths.ntu) + 1) / len(lengths.ntu)
        hml_cdf = np.arange(1, len(lengths.hml) + 1) / len(lengths.hml)
        
        axes[2].plot(lengths.ntu, ntu_cdf, label='NTU', color=self.color['NTU'])
        axes[2].plot(lengths.hml, hml_cdf, label='HML', color=self.color['HML'])
        axes[2].set_xlabel('Sequence Length')
        axes[2].set_ylabel('Cumulative Probability')
        axes[2].set_title('Cumulative Distribution Function')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(pjoin(self.out_dir, 'sequence_length_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_statistics(self):
        """Plot feature-wise statistics comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        axes[0].plot(self.stats.ntu['global']['mean'].flatten(), label='NTU', color=self.color['NTU'], alpha=0.7)
        axes[0].plot(self.stats.hml['global']['mean'].flatten(), label='HML', color=self.color['HML'], alpha=0.7)
        axes[0].set_xlabel('Feature Dimension')
        axes[0].set_ylabel('Mean Value')
        axes[0].set_title('Feature-wise Mean Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Standard deviation comparison
        axes[1].plot(self.stats.ntu['global']['std'].flatten(), label='NTU', color=self.color['NTU'], alpha=0.7)
        axes[1].plot(self.stats.hml['global']['std'].flatten(), label='HML', color=self.color['HML'], alpha=0.7)
        axes[1].set_xlabel('Feature Dimension')
        axes[1].set_ylabel('Standard Deviation')
        axes[1].set_title('Feature-wise Std Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        def plot_feat_range(data_stats, data_name):
            max_feat = data_stats['global']['max'].flatten()
            min_feat = data_stats['global']['min'].flatten()
            # Plot upper bounds (max values)
            x_ticks = np.arange(self.num_feats)
            axes[2].plot(x_ticks, max_feat, alpha=0.8, color=self.color[data_name], linestyle='-', linewidth=1.5)
            axes[2].plot(x_ticks, min_feat, alpha=0.8, color=self.color[data_name], linestyle='-', linewidth=1.5)            
            # Fill between min and max to show the range visually
            axes[2].fill_between(x_ticks, min_feat, max_feat, alpha=0.1, color=self.color[data_name], label=f'{data_name} Range')
        
        plot_feat_range(self.stats.ntu, 'NTU')
        plot_feat_range(self.stats.hml, 'HML')

        axes[2].set_xlabel('Feature Dimension')
        axes[2].set_ylabel('Feature Value')
        axes[2].set_title('Feature-wise Min/Max Bounds Comparison')
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(pjoin(self.out_dir, 'feature_statistics.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def motion_analysis(self, percentile=95.0, iqr_scale=1.5):
        """Analyze motion characteristics"""
        assert self.args.data_rep == 'xyz', "Motion analysis requires data in XYZ format (T, J, 3)"
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        
        # Extract Velocity and Relative Magnitudes
        ntu_vel, ntu_acc, ntu_jerk = motion_stats_from_xyz(self.data.ntu)
        hml_vel, hml_acc, hml_jerk = motion_stats_from_xyz(self.data.hml)        

        ##
        ## MOTION SEQUENCE-WISE ANALYSIS
        ##

        # unified outlier detection
        blacklist = {'ntu':set(), 'hml':set()}

        def plot_sequence_stats(ntu_data, hml_data, axes, idx, ylabel, title, metric_name):
            """Plot sequence-wise statistics"""
            # Average magnitude per sequence
            ntu_seq_avg = [np.mean(seq) for seq in ntu_data]
            hml_seq_avg = [np.mean(seq) for seq in hml_data]
            
            def get_outliers_by_iqr(data):
                """Get indices of IQR outliers (same as boxplot dots)"""
                Q1, Q3 = np.percentile(data, [25, 75])
                IQR = Q3 - Q1
                outliers = np.where((data < Q1 - iqr_scale*IQR) | (data > Q3 + iqr_scale*IQR))[0]
                fence_dist = lambda i: (Q1 - iqr_scale*IQR - data[i]) if data[i] < Q1 - iqr_scale*IQR else (data[i] - Q3 - iqr_scale*IQR)
                return sorted(outliers, key=fence_dist, reverse=True)

            def store_outliers_filenames(seq_avg, dataset_name):
                """Store IQR outlier filenames"""
                outliers = get_outliers_by_iqr(seq_avg)
                fnames = getattr(self.filenames, dataset_name)
                with open(pjoin(self.outliers_dir, f'{dataset_name}_{metric_name}.txt'), 'w') as f:
                    for i in outliers:
                        blacklist[dataset_name].add(fnames[i])
                        f.write(f"{fnames[i]}\n")

            store_outliers_filenames(ntu_seq_avg, 'ntu')
            #store_outliers_filenames(hml_seq_avg, 'hml')

            # plot
            axes[0, idx].boxplot([ntu_seq_avg, hml_seq_avg], labels=['NTU', 'HML'], whis=iqr_scale)
            axes[0, idx].set_ylabel(ylabel)
            axes[0, idx].set_title(title)
            axes[0, idx].axhline(np.percentile(ntu_seq_avg, percentile), color=self.color['NTU'], linestyle='--', linewidth=0.5, label=f'NTU {percentile}%')
            axes[0, idx].axhline(np.percentile(hml_seq_avg, percentile), color=self.color['HML'], linestyle='--', linewidth=0.5, label=f'HML {percentile}%')
            axes[0, idx].legend()

        plot_sequence_stats(ntu_vel['mag'], hml_vel['mag'], axes, 0, 'Avg. Velocity Magnitude', 'Average Velocity per Sequence', 'velocity')
        plot_sequence_stats(ntu_acc['mag'], hml_acc['mag'], axes, 1, 'Avg. Acceleration Magnitude', 'Average Acceleration per Sequence', 'acceleration')
        plot_sequence_stats(ntu_jerk['mag'], hml_jerk['mag'], axes, 2, 'Avg. Jerk Magnitude', 'Average Jerk per Sequence (Motion Smoothness)', 'jerk')

        def write_blacklist(dataset_name):            
            """Write blacklist to file"""
            with open(pjoin(self.outliers_dir, f'{dataset_name}_blacklist.txt'), 'w') as f:
                for x in blacklist[dataset_name]:
                    f.write(f"{x}\n")

        write_blacklist('ntu')
        #write_blacklist('hml')

        ##
        ## FRAME-WISE ANALYSIS
        ##

        def plot_frame_stats(ntu_data, hml_data, axes, idx, ylabel, title):
            """Plot frame-wise statistics"""
            ntu_flat = np.concatenate([seq.flatten() for seq in ntu_data], axis=0)
            hml_flat = np.concatenate([seq.flatten() for seq in hml_data], axis=0)
            # Calculate thresholds for outlier detection
            ntu_thresh = np.percentile(ntu_flat, percentile)
            hml_thresh = np.percentile(hml_flat, percentile)
            # Filter data below thresholds for better visualization
            ntu_data = ntu_flat[ntu_flat < ntu_thresh]
            hml_data = hml_flat[hml_flat < hml_thresh]
            # plot
            axes[1, idx].hist(ntu_data, bins=100, alpha=0.7, label='NTU', color=self.color['NTU'], density=True)
            axes[1, idx].hist(hml_data, bins=100, alpha=0.7, label='HML', color=self.color['HML'], density=True)
            axes[1, idx].set_xlabel('Magnitude')
            axes[1, idx].set_ylabel(ylabel)
            axes[1, idx].set_title(title)
            axes[1, idx].legend()

        plot_frame_stats(ntu_vel['mag'], hml_vel['mag'], axes, 0, 'Density', f'Velocity Magnitude Distribution per Frame ({percentile}th percentile)')
        plot_frame_stats(ntu_acc['mag'], hml_acc['mag'], axes, 1, 'Density', f'Acceleration Magnitude Distribution per Frame ({percentile}th percentile)')
        plot_frame_stats(ntu_jerk['mag'], hml_jerk['mag'], axes, 2, 'Density', f'Jerk Magnitude Distribution per Frame ({percentile}th percentile)')
        
        plt.tight_layout()
        plt.savefig(pjoin(self.out_dir, 'motion_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def statistical_tests(self):
        """Perform statistical tests to compare datasets"""
        logging.info("\n" + "="*50)
        logging.info("STATISTICAL SIGNIFICANCE TESTS")
        logging.info("="*50)
        
        # Compare sequence lengths
        ntu_lengths = [seq.shape[0] for seq in self.data.ntu]
        hml_lengths = [seq.shape[0] for seq in self.data.hml]
    
        # Kolmogorov-Smirnov test for sequence length distributions
        ks_stat, ks_p = stats.ks_2samp(ntu_lengths, hml_lengths)
        logging.info(f"\nSequence Length Distribution (Kolmogorov-Smirnov Test):")
        logging.info(f"KS Statistic: {ks_stat}")
        logging.info(f"P-value: {ks_p}")
        logging.info(f"Distributions significantly different: {'Yes' if ks_p < 0.05 else 'No'}")
        
        # Compare feature ranges
        ntu_mean_range = np.mean(self.stats.ntu['global']['max'] - self.stats.ntu['global']['min'])
        hml_mean_range = np.mean(self.stats.hml['global']['max'] - self.stats.hml['global']['min'])
        
        logging.info(f"\nFeature Range Comparison:")
        logging.info(f"NTU {ntu_mean_range:.4f}")
        logging.info(f"HML {hml_mean_range:.4f}")
        logging.info(f"Ratio (NTU/HML): {ntu_mean_range/hml_mean_range:.4f}")
        
       
    def dimensionality_analysis(self, sample_size=10000, window_size=20, percentile=100.0):
        """Dimensionality analysis with joint PCA and 3D visualization"""
        # Random sampling
        sample_size = min(sample_size, len(self.data.ntu), len(self.data.hml))
        ntu_sample = random.sample([m for m in self.data.ntu if m.shape[0] >= window_size], sample_size)
        hml_sample = random.sample([m for m in self.data.hml if m.shape[0] >= window_size], sample_size)
        logging.info(f"Sample size: {len(ntu_sample) + len(hml_sample)} (NTU: {len(ntu_sample)}, HML: {len(hml_sample)})")

        # Ensure all samples have the same number of frames
        window_size = min(min([m.shape[0] for m in ntu_sample]), min([m.shape[0] for m in hml_sample]), window_size)
        logging.info(f"Window size: {window_size}")
        ntu_sample = [m[:window_size] for m in ntu_sample]
        hml_sample = [m[:window_size] for m in hml_sample]
        
        # unify over frame dimension
        ntu_sample = np.concatenate(ntu_sample, axis=0)
        hml_sample = np.concatenate(hml_sample, axis=0)

        # Combine datasets for joint PCA
        combined = np.vstack([ntu_sample, hml_sample])
        if self.args.data_rep == 'xyz':
            combined = combined.reshape(combined.shape[0], -1)  # (T, J, 3) -> (T, J*3)
        space_dim = combined.shape[-1]

        # Scale combined data
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)
        # PCA on combined scaled data
        pca = PCA()
        combined_pca = pca.fit_transform(combined_scaled)
        var_ratio = pca.explained_variance_ratio_
        # Split back into NTU and HML projected data
        ntu_pca = combined_pca[:ntu_sample.shape[0]]
        hml_pca = combined_pca[ntu_sample.shape[0]:]
        
        # Plotting
        _ = plt.figure(figsize=(20, 12))
        
        # Cumulative variance explained
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(np.cumsum(var_ratio), label='Combined Data', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Cumulative Variance Explained')
        ax1.set_title('PCA: Cumulative Variance Explained (Joint PCA)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Individual variance explained for first components
        n_comp = min(50, len(var_ratio))
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(var_ratio[:n_comp], label='Combined Data', alpha=0.8, linewidth=2)
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Variance Explained')
        ax2.set_title(f'Individual Variance Explained\n(First {n_comp} Components)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sample points for visualization to avoid overcrowding
        n_vis_points = 2000
        if ntu_pca.shape[0] > n_vis_points:
            ntu_vis_idx = np.random.choice(ntu_pca.shape[0], n_vis_points, replace=False)
            ntu_vis = ntu_pca[ntu_vis_idx]
        else:
            ntu_vis = ntu_pca
        
        if hml_pca.shape[0] > n_vis_points:
            hml_vis_idx = np.random.choice(hml_pca.shape[0], n_vis_points, replace=False)
            hml_vis = hml_pca[hml_vis_idx]
        else:
            hml_vis = hml_pca
        
        # 3D PCA visualization
        ax3 = plt.subplot(2, 3, 3, projection='3d')
        ax3.scatter(ntu_vis[:, 0], ntu_vis[:, 1], ntu_vis[:, 2], c=self.color['NTU'], alpha=0.6, s=1, label='NTU')
        ax3.scatter(hml_vis[:, 0], hml_vis[:, 1], hml_vis[:, 2], c=self.color['HML'], alpha=0.6, s=1, label='HML')
        ax3.set_xlabel(f'PC1 ({var_ratio[0]:.1%} var)')
        ax3.set_ylabel(f'PC2 ({var_ratio[1]:.1%} var)')
        ax3.set_zlabel(f'PC3 ({var_ratio[2]:.1%} var)')
        ax3.set_title('3D PCA Visualization\n(First 3 Components)')
        ax3.legend()
        subplot_start = 4
        
        # 2D projections
        ax4 = plt.subplot(2, 3, subplot_start)
        ax4.scatter(ntu_vis[:, 0], ntu_vis[:, 1], c='blue', alpha=0.5, s=1, label='NTU')
        ax4.scatter(hml_vis[:, 0], hml_vis[:, 1], c='red', alpha=0.5, s=1, label='HML')
        ax4.set_xlabel(f'PC1 ({var_ratio[0]:.1%} var)')
        ax4.set_ylabel(f'PC2 ({var_ratio[1]:.1%} var)')
        ax4.set_title('PC1 vs PC2')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(2, 3, subplot_start + 1)
        ax5.scatter(ntu_vis[:, 0], ntu_vis[:, 2], c='blue', alpha=0.5, s=1, label='NTU')
        ax5.scatter(hml_vis[:, 0], hml_vis[:, 2], c='red', alpha=0.5, s=1, label='HML')
        ax5.set_xlabel(f'PC1 ({var_ratio[0]:.1%} var)')
        ax5.set_ylabel(f'PC3 ({var_ratio[2]:.1%} var)')
        ax5.set_title('PC1 vs PC3')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(2, 3, subplot_start + 2)
        ax6.scatter(ntu_vis[:, 1], ntu_vis[:, 2], c='blue', alpha=0.5, s=1, label='NTU')
        ax6.scatter(hml_vis[:, 1], hml_vis[:, 2], c='red', alpha=0.5, s=1, label='HML')
        ax6.set_xlabel(f'PC2 ({var_ratio[1]:.1%} var)')
        ax6.set_ylabel(f'PC3 ({var_ratio[2]:.1%} var)')
        ax6.set_title('PC2 vs PC3')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(pjoin(self.out_dir, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        logging.info("\n" + "="*60)
        logging.info("COMPREHENSIVE DATASET COMPARISON REPORT")
        logging.info("="*60)
        
        logging.info("\n1. DATASET OVERVIEW:")
        logging.info(f"   NTU Dataset: {len(self.data.ntu)} sequences")
        logging.info(f"   HML Dataset: {len(self.data.hml)} sequences")
        logging.info(f"   NTU Total Frames: {self.stats.ntu['global']['total_frames']:,}")
        logging.info(f"   HML Total Frames: {self.stats.hml['global']['total_frames']:,}")
        
        logging.info("\n2. SEQUENCE CHARACTERISTICS:")
        ntu_seq = self.stats.ntu['lengths']
        hml_seq = self.stats.hml['lengths']
        logging.info(f"   NTU - Mean Length: {ntu_seq['mean']:.1f} ± {ntu_seq['std']:.1f} frames")
        logging.info(f"         Range: {ntu_seq['min']}-{ntu_seq['max']} frames")
        logging.info(f"   HML - Mean Length: {hml_seq['mean']:.1f} ± {hml_seq['std']:.1f} frames")
        logging.info(f"         Range: {hml_seq['min']}-{hml_seq['max']} frames")
        
        logging.info("\n4. MOTION CHARACTERISTICS:")
        if 'motion_stats' in self.stats.ntu and 'motion_stats' in self.stats.hml:
            logging.info(f"   NTU - Avg Velocity Magnitude: {self.stats.ntu['motion_stats']['velocity_magnitude_mean']:.4f}")
            logging.info(f"   HML - Avg Velocity Magnitude: {self.stats.hml['motion_stats']['velocity_magnitude_mean']:.4f}")
                
        # Save summary to file
        logging.info("Motion Dataset Comparison Summary\n")
        logging.info("="*40 + "\n")
        logging.info("Run MetaData:\n")
        for name, value in vars(self.args).items():
            logging.info(f"\t{name} : {value}")
        logging.info("="*40 + "\n\n")
        logging.info(f"NTU Dataset: {len(self.data.ntu)} sequences, {self.stats.ntu['global']['total_frames']:,} frames\n")
        logging.info(f"HML Dataset: {len(self.data.hml)} sequences, {self.stats.hml['global']['total_frames']:,} frames\n\n")
        logging.info(f"NTU Mean Sequence Length: {ntu_seq['mean']:.1f} ± {ntu_seq['std']:.1f}\n")
        logging.info(f"HML Mean Sequence Length: {hml_seq['mean']:.1f} ± {hml_seq['std']:.1f}\n")
            

    def run_full_analysis(self, outlier_cutoff=1.5):
        """Run the complete analysis pipeline"""
        
        logging.info("\nStarting comprehensive motion dataset analysis...")
        self.load_datasets()
        if len(self.data.ntu) == 0 or len(self.data.hml) == 0:
            logging.info("Error: One or both datasets are empty. Please check the file paths.")
            return

        # Compute statistics
        self.compute_basic_statistics()
        logging.info("\nPlotting basic stats... (lenght distribution, per-feature global stats)")
        self.plot_sequence_length_distribution()
        self.plot_feature_statistics()

        logging.info("\nBeginning statistical tests...")
        self.statistical_tests()

        if self.args.data_rep == 'xyz':
            logging.info("\nBeginning motion dynamics analysis...")
            self.motion_analysis(iqr_scale=outlier_cutoff)
        
        logging.info("\nBeginning dimensionality analysis...")
        self.dimensionality_analysis()

        self.generate_summary_report()
        logging.info("\nAnalysis complete! Check generated plots and summary file.")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis (EDA) for NTU RGB+D vs. HumanML3D Datasets")
    parser.add_argument('--dataset', default='NTU60', type=str, choices=['NTU60', 'NTU120'], help='Which NTU dataset to use')
    parser.add_argument('--data-rep', default='xyz', type=str, required=True, choices=['xyz', 'hml_vec'], help='data representation format')
    parser.add_argument('--hml-set', default='', type=str, help='Path to .txt listing a specific set of HML filenames to analyze, if empty all files are used')
    parser.add_argument('--ntu-set', default='', type=str, help='Path to .txt listing a specific set of NTU filenames to analyze, if empty all files are used')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--outlier-cutoff', default=1.5, type=float, help='Scale of IQR, used for outlier detection')
    parser.add_argument('--use-cache', action='store_true', help='Use cached data if available')
    args = parser.parse_args()

    if args.data_rep == 'hml_vec':
        data_folder = 'new_joint_vecs'
    elif args.data_rep == 'xyz':
        data_folder = 'new_joints'
    else:
        raise ValueError("Unrecognized data representation format : {}".format(args.data_rep))
    
    ROOT = Path('.').resolve()
    OUT_PATH = pjoin(Path(__file__).parent.relative_to(ROOT), 'outputs', args.dataset)
    NTU_DIR = pjoin(ROOT, 'data', args.dataset, data_folder)
    HML_DIR = pjoin(ROOT, 'data', 'HumanML3D', data_folder)
    CACHE_DIR = 'cache'

    # Initialize analyzer
    analyzer = MotionDatasetAnalyzer(NTU_DIR, HML_DIR, CACHE_DIR, OUT_PATH, args)
    
    # Run complete analysis
    analyzer.run_full_analysis(outlier_cutoff=args.outlier_cutoff)
    
    # Individual analysis components can also be run separately:
    # analyzer.load_datasets()
    # analyzer.compute_basic_statistics()
    # analyzer.plot_sequence_length_distribution()
    # analyzer.statistical_tests()