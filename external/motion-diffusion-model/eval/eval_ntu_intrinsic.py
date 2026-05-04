import torch
import numpy as np
import os
import time
from collections import OrderedDict
from datetime import datetime
from scipy import linalg

# Standard imports
from utils import dist_util
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from data_loaders.get_data import get_single_stream_dataloader
from eval.evaluators.evaluator_wrapper import EvaluatorWrapper
from eval.evaluators.metrics import calculate_activation_statistics, calculate_frechet_distance

# Multiprocessing setup
torch.multiprocessing.set_sharing_strategy('file_system')

def calculate_diversity_robust(activation, diversity_times):
    """
    Robust diversity calculation that handles small datasets (like few-shot sets)
    by computing exact pairwise distances instead of random sampling.
    """
    num_samples = activation.shape[0]
    
    # CASE 1: Small Dataset (e.g., < diversity_times)
    # Use exact mean pairwise distance of all unique pairs.
    if num_samples < diversity_times:
        # Compute difference matrix: (N, 1, D) - (1, N, D) -> (N, N, D)
        diff = activation[:, None, :] - activation[None, :, :]
        
        # Compute Euclidean norm along the feature dimension
        dist_matrix = np.linalg.norm(diff, axis=-1)
        
        # The matrix includes 0s on the diagonal (distance to self).
        # We sum all distances and divide by N*(N-1) to get the average distance between distinct pairs.
        if num_samples > 1:
            total_dist = dist_matrix.sum()
            n_pairs = num_samples * (num_samples - 1)
            return total_dist / n_pairs
        else:
            return 0.0

    # CASE 2: Large Dataset (Standard Behavior)
    else:
        first_indices = np.random.choice(num_samples, diversity_times, replace=False)
        second_indices = np.random.choice(num_samples, diversity_times, replace=False)
        dist = np.linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
        return dist.mean()

def get_activations(eval_wrapper, motion_loaders):
    """
    Extracts features (embeddings) for the given loaders.
    """
    activation_dict = OrderedDict({})
    for motion_loader_name, motion_loader in motion_loaders.items():
        # print(f"--> Extracting features for [{motion_loader_name}]") 
        # (Reduced print verbosity for the loop)
        out = {'embedding' : []}
        
        loader_iter = motion_loader() if callable(motion_loader) else motion_loader

        with torch.no_grad():
            for batch in loader_iter:
                if len(batch) >= 3:
                    motions, m_lens = batch[0], batch[1]
                else:
                    raise ValueError("Unexpected batch format in loader")

                motions = motions.to(eval_wrapper.device)
                m_lens = m_lens.to(eval_wrapper.device)

                embedding, _ = eval_wrapper.get_motion_embeddings(
                    x=motions,
                    y={'lengths':m_lens}
                )

                out['embedding'].append(embedding.cpu().numpy())
            
            out['embedding'] = np.concatenate(out['embedding'], axis=0)

        activation_dict[motion_loader_name] = out
    
    return activation_dict

def evaluate_baseline_metrics(eval_wrapper, gt_loader, fs_loader, log_file, replication_times=5, diversity_times=300):
    with open(log_file, 'w') as f:    
        
        print(f'========== STARTING REAL-TO-REAL BASELINE EVALUATION ==========', file=f, flush=True)
        print(f'Comparing: Few-Shot Training Set vs. Full Validation Set', file=f, flush=True)
        print(f'Replications: {replication_times}', file=f, flush=True)

        all_metrics = OrderedDict({
            'FID': [],
            'Diversity_FewShot': [],
            'Diversity_Validation': [],
        })

        for replication in range(replication_times):
            print(f'\n==================== Run {replication+1}/{replication_times} ====================')
            print(f'\n==================== Run {replication+1}/{replication_times} ====================', file=f, flush=True)

            # 1. Get Few-Shot Features (Re-extracted each time)
            fs_dict = get_activations(eval_wrapper, {'FewShot_Train_Set': fs_loader})
            fs_embeddings = fs_dict['FewShot_Train_Set']['embedding']

            # 2. Get Validation Features (Re-extracted each time)
            gt_dict = get_activations(eval_wrapper, {'Validation_Set': gt_loader})
            gt_embeddings = gt_dict['Validation_Set']['embedding']

            # --- Calculate Stats ---
            fs_mu, fs_cov = calculate_activation_statistics(fs_embeddings)
            gt_mu, gt_cov = calculate_activation_statistics(gt_embeddings)

            # --- Metric: FID ---
            print('--- Calculating FID ---')
            fid = calculate_frechet_distance(gt_mu, gt_cov, fs_mu, fs_cov)
            print(f'---> FID: {fid:.4f}')
            print(f'---> FID: {fid:.4f}', file=f, flush=True)
            all_metrics['FID'].append(fid)

            # --- Metric: Diversity (FewShot) ---
            print('--- Calculating Diversity (FewShot) ---')
            div_fs = calculate_diversity_robust(fs_embeddings, diversity_times)
            print(f'---> Diversity (FS): {div_fs:.4f}')
            print(f'---> Diversity (FS): {div_fs:.4f}', file=f, flush=True)
            all_metrics['Diversity_FewShot'].append(div_fs)

            # --- Metric: Diversity (Validation) ---
            print('--- Calculating Diversity (Validation) ---')
            div_val = calculate_diversity_robust(gt_embeddings, diversity_times)
            print(f'---> Diversity (Val): {div_val:.4f}')
            print(f'---> Diversity (Val): {div_val:.4f}', file=f, flush=True)
            all_metrics['Diversity_Validation'].append(div_val)

        # SUMMARY
        print('\n========== FINAL BASELINE RESULTS ==========')
        print('\n========== FINAL BASELINE RESULTS ==========', file=f, flush=True)
        for metric, values in all_metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            conf_interval = 1.96 * std / np.sqrt(replication_times)
            
            print(f'{metric}: Mean {mean:.4f} ± {conf_interval:.4f} (Std: {std:.4f})')
            print(f'{metric}: Mean {mean:.4f} ± {conf_interval:.4f} (Std: {std:.4f})', file=f, flush=True)

if __name__ == "__main__":
    args = evaluation_parser()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    
    stream = args.single_stream if args.model_type in ['MDM'] else 'target'
    stream_args = getattr(args, stream)

    print(f"\n>>> Initializing Real-to-Real Baseline Evaluation for [{stream_args.dataset}] <<<")

    print(f"Loading EvaluatorWrapper...")
    eval_wrapper = EvaluatorWrapper(
        stream_args.dataset, 
        dist_util.dev(), 
        task_split=stream_args.task_split, 
        fewshot_id=stream_args.fewshot_id
    )

    print(f"Creating Full Validation Loader...")
    val_loader = get_single_stream_dataloader(
        data_stream_args=stream_args,
        batch_size=args.batch_size,
        split='val',        
        hml_mode='gt',      
        evaluator='stgcn', 
        data_rep='xyz',
        device=dist_util.dev(),
    )

    print(f"Creating Few-Shot Training Loader...")
    fs_loader = get_single_stream_dataloader(
        data_stream_args=stream_args,
        batch_size=args.batch_size,
        split='train',      
        hml_mode='gt',      
        evaluator='stgcn',
        data_rep='xyz',
        device=dist_util.dev(),
    )

    print(f"Validation Samples: {len(val_loader.dataset)}")
    print(f"Few-Shot Samples: {len(fs_loader.dataset)}")

    log_name = f'eval_baseline_FS_vs_VAL_{stream_args.dataset}.log'
    log_path = os.path.join(os.path.dirname(args.model_path), log_name)
    
    # Standard replication times set to 5 as requested
    evaluate_baseline_metrics(
        eval_wrapper, 
        val_loader, 
        fs_loader, 
        log_path, 
        replication_times=5, 
        diversity_times=300
    )