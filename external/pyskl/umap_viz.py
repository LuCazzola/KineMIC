import sys

# --- PYTHON 3.7 COMPATIBILITY PATCH ---
if sys.version_info < (3, 8):
    try:
        import importlib_metadata
        sys.modules['importlib.metadata'] = importlib_metadata
    except ImportError:
        pass 
# --------------------------------------

import argparse
import os
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # Needed for custom legend

try:
    import umap
except ImportError:
    import umap.umap_ as umap

from pyskl.datasets import build_dataloader, build_dataset
from pyskl.models import build_model
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel, DataContainer

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PySKL model with Clean Class-Colored UMAP.')
    parser.add_argument('config', help='Path to the python config file')
    parser.add_argument('checkpoint', help='Path to the .pth checkpoint file')
    parser.add_argument('--split', default='val', help='The target split to evaluate (val/test)')
    parser.add_argument('--max_train_samples', type=int, default=2000, 
                        help='Max samples to load from training set (saves time).')
    return parser.parse_args()

# --- FEATURE EXTRACTION ---
def extract_features_and_labels(model, data_loader, device, max_samples=None):
    model.eval()
    
    temp_feats = []
    def hook_fn(module, input, output):
        feat = input[0].detach().cpu()
        temp_feats.append(feat)
    
    handle = model.module.cls_head.register_forward_hook(hook_fn)

    print(f"   > Extracting features (Limit: {max_samples if max_samples else 'All'})...")
    
    labels_list = []
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            
            kp_tensor = data['keypoint']
            lbl = data['label'].item()
            labels_list.append(lbl)

            if kp_tensor.ndim == 4:
                kp_tensor = kp_tensor.unsqueeze(1)
            
            kp_tensor = kp_tensor.to(device)
            _ = model(kp_tensor, return_loss=False)
            
            if max_samples and (i + 1) >= max_samples:
                break
    
    handle.remove()

    final_feats = []
    for f in temp_feats:
        if f.dim() > 2:
            f = f.view(f.size(0), f.size(1), -1).mean(dim=-1)
        final_feats.append(f.numpy())

    return np.concatenate(final_feats), np.array(labels_list)

# --- CLEAN PLOTTING FUNCTION ---
def plot_clean_umap(train_emb, train_lbl, val_emb, val_lbl, class_names, save_path):
    print("   > Generating clean UMAP plot...")
    
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('tab10')
    unique_classes = np.unique(np.concatenate([train_lbl, val_lbl]))
    
    # Lists to build the custom legend
    legend_handles = []
    legend_labels = []

    # 1. Plot Data by Class
    for i, cls_idx in enumerate(unique_classes):
        label_name = class_names[cls_idx] if class_names else f"Class {cls_idx}"
        color = cmap(i / len(unique_classes) if len(unique_classes) > 10 else i)
        
        # Filter Data
        train_mask = train_lbl == cls_idx
        val_mask = val_lbl == cls_idx
        
        # Plot Train (Increased Alpha for visibility)
        plt.scatter(train_emb[train_mask, 0], train_emb[train_mask, 1],
                    c=[color], marker='o', alpha=0.5, s=40,
                    linewidths=0) # No edge for train to keep it "background-like"
        
        # Plot Test
        plt.scatter(val_emb[val_mask, 0], val_emb[val_mask, 1],
                    c=[color], marker='^', alpha=0.9, s=70, 
                    edgecolors='k', linewidth=0.5)

        # Add Class Color to Legend
        legend_handles.append(Line2D([0], [0], marker='s', color='w', 
                                     markerfacecolor=color, markersize=10))
        legend_labels.append(label_name)

    # 2. Add Shape Explanations to Legend
    # Add a spacer (empty handle)
    legend_handles.append(Line2D([0], [0], color='w', label=''))
    legend_labels.append("_________________")
    
    # Train Marker
    legend_handles.append(Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='gray', alpha=0.6, markersize=10))
    legend_labels.append("Train Data (Circle)")
    
    # Test Marker
    legend_handles.append(Line2D([0], [0], marker='^', color='w', 
                                 markerfacecolor='gray', alpha=0.9, markersize=10,
                                 markeredgecolor='k'))
    legend_labels.append("Test Data (Triangle)")

    # 3. Finalize Plot
    plt.title('UMAP Feature Projection', fontsize=18)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # Place Legend outside top right
    plt.legend(legend_handles, legend_labels, 
               bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
               fontsize=11)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ Plot saved to: {save_path}")

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("--- Building Model ---")
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0]).to(device)
    model.eval()

    # Datasets
    print(f"\n--- Loading Training Data (Subset: {args.max_train_samples}) ---")
    cfg.data.train.test_mode = True 
    train_dataset = build_dataset(cfg.data.train, dict(test_mode=True))
    train_loader = build_dataloader(train_dataset, videos_per_gpu=1, workers_per_gpu=2, shuffle=True)

    print(f"\n--- Loading {args.split.capitalize()} Data ---")
    val_dataset = build_dataset(cfg.data[args.split], dict(test_mode=True))
    val_loader = build_dataloader(val_dataset, videos_per_gpu=1, workers_per_gpu=2, shuffle=False)

    # Features
    print("\nExtracting Training Features...")
    train_feats, train_lbls = extract_features_and_labels(model, train_loader, device, max_samples=args.max_train_samples)
    print("\nExtracting Validation Features...")
    val_feats, val_lbls = extract_features_and_labels(model, val_loader, device, max_samples=None)

    target_names = getattr(val_dataset, 'CLASSES', None)

    # UMAP
    print("\n--- Computing UMAP ---")
    reducer = umap.UMAP(
        n_neighbors=150, 
        min_dist=0.001, 
        n_components=2, 
        metric='euclidean',  # Switched back to euclidean as requested
        random_state=42
    )
    
    print("   > Fitting UMAP...")
    embedding_train = reducer.fit_transform(train_feats)
    embedding_val = reducer.transform(val_feats)

    # Plot
    save_file = f"clean_umap_{os.path.basename(args.checkpoint).replace('.pth', '')}.png"
    plot_clean_umap(embedding_train, train_lbls, 
                    embedding_val, val_lbls, 
                    target_names, save_file)

if __name__ == '__main__':
    main()