import os
import json
import argparse

from pathlib import Path
from os.path import join as pjoin
import numpy as np
import json
import torch
from tqdm import tqdm
import clip
from sklearn.neighbors import NearestNeighbors
import random
import pandas as pd

def load_texts(root_folder, samples):
    descriptions = []
    for file_path in tqdm([os.path.join(root_folder, f'{s}.txt') for s in samples]):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('#')[0].strip() for line in f if line.strip()]
            descriptions.append(random.choice(lines)) # Randomly select one description sample
    return descriptions

def encode_texts(text_list, model, device, batch_size=32):
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(text_list))
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())
            pbar.update(len(batch))
        pbar.close()
    return torch.cat(all_embeddings, dim=0)

def get_text_embedding(text_list, model, device, args, use_cache=False, out_file=''):
    cached_embeddings = os.path.join(args.cache_dir, out_file)
    if use_cache and out_file != '' and os.path.exists(cached_embeddings):
        print(f"Loading cached embeddings from: {cached_embeddings}")
        return torch.load(cached_embeddings, map_location='cpu')
    else:
        print(f"Computing embeddings and saving to: {cached_embeddings}")
        embeddings = encode_texts(text_list, model, device)
        os.makedirs(args.cache_dir, exist_ok=True)
        torch.save(embeddings, cached_embeddings)
        return embeddings

def compute_knn_stats(
    ntu_embeds, ntu_labels, ntu_texts,
    hml_embeds, hml_texts, args, out_path=''
):
    print(f"Computing crowding using K={args.k} nearest neighbors...")
    os.makedirs(args.save_dir, exist_ok=True)

    nn = NearestNeighbors(n_neighbors=args.k, metric='cosine')
    nn.fit(hml_embeds)

    label_stats = []
    TOP_K = 5
    
    for idx, emb in enumerate(ntu_embeds):
        emb = emb.unsqueeze(0)
        distances, indices = nn.kneighbors(emb, return_distance=True)
        distances = distances.flatten()
        indices = indices.flatten()

        kth_distance = distances[-1]
        # Fixed: Higher crowding = lower average distance (more similar to HumanML3D)
        crowding = 1.0 / (1.0 + kth_distance) if kth_distance >= 0 else 0.0

        # Fixed: Sort by distance only to get closest neighbors first
        sorted_pairs = sorted(zip(distances, indices), key=lambda x: x[0])
        top_k_similarities = [
            {"distance": float(d), "text": hml_texts[i]} 
            for d, i in sorted_pairs[:TOP_K]
        ]

        label_stats.append({
            "action": ntu_labels[idx],
            "action_label": ntu_texts[idx],
            "text": ntu_texts[idx],
            "crowding_score": crowding,
            f"top-{TOP_K}-sim": top_k_similarities
        })
    
    # Final JSON structure
    result = {
        "k": args.k,
        "labels": label_stats
    }
    
    # Optionally save to a file
    out_file = os.path.join(out_path, args.save_dir, f"ntu_crowding_k{args.k}_{'action-label-only' if args.use_action_label else ''}.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=4)

    return result, out_file

def rank_ntu_labels_by_crowding(data):
    """Rank NTU labels by their crowding scores."""

    labels_data = data["labels"]

    # Build DataFrame
    data = []
    for item in labels_data:
        # Handle both old and new field names for backward compatibility
        crowding_score = item.get("crowding_score", 0)
        data.append({
            "rank": None,
            "action_id": item["action"],
            "action_label": item.get("action_label", item.get("text", "Unknown")),
            "crowding_score": crowding_score
        })
    
    df = pd.DataFrame(data)
    
    # Sort by crowding score (higher is better) and reset index
    df = df.sort_values(by="crowding_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # Configure pandas display options for better formatting
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.colheader_justify", "center")
    pd.set_option("display.float_format", "{:.6f}".format)

    print("\n> NTU Labels ranked by estimated crowding (higher = more similar to HumanML3D):\n")
    print(df[["rank", "action_id", "action_label", "crowding_score"]].to_string(index=False))
    
    # Print some summary statistics
    print(f"\n> Summary Statistics:")
    print(f"   Total labels: {len(df)}")
    print(f"   Highest crowding: {df['crowding_score'].max():.6f} ({df.iloc[0]['action_label']})")
    print(f"   Lowest crowding: {df['crowding_score'].min():.6f} ({df.iloc[-1]['action_label']})")
    print(f"   Mean crowding: {df['crowding_score'].mean():.6f}")
    print(f"   Median crowding: {df['crowding_score'].median():.6f}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute k-NN stats between NTU actions and HumanML3D descriptions.")
    parser.add_argument("--k", type=int, default=500, help="Number of nearest neighbors to consider.")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Directory to save cached embeddings.")
    parser.add_argument("--use-cache", action="store_true", default=False, help="Whether to use cached embeddings if available (default: False).")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save results.")
    parser.add_argument("--use-action-label", action="store_true", default=False, help="Use action label directly as text prompt for action classes, otherwise use natural language adaptation (default: False).")
    args = parser.parse_args()

    ROOT = Path('.').resolve()
    OUT_PATH = Path(__file__).parent.relative_to(ROOT)
    
    SPLIT = 'train'
    NTU_TASK = 'xsub'
    DATA_ROOT = {
        'NTU' : pjoin(ROOT, 'data', 'NTU60'),
        'HML' : pjoin(ROOT, 'data', 'HumanML3D'),
    }
    TXT_ROOT = {
        'NTU' : pjoin(DATA_ROOT['NTU'], 'texts'),
        'HML' : pjoin(DATA_ROOT['HML'], 'texts'),
    }

    CACHE_DIR = pjoin(ROOT, args.cache_dir)
    SAVE_DIR = pjoin(ROOT, args.save_dir)
    
    # Load data
    with open(pjoin(DATA_ROOT['NTU'], 'splits', 'default', NTU_TASK, f'{SPLIT}.txt'), 'r', encoding='utf-8') as f:
        ntu60_samples = [line.strip() for line in f]
    with open(pjoin(DATA_ROOT['NTU'], 'splits', 'default', NTU_TASK, f'{SPLIT}_y.txt'), 'r', encoding='utf-8') as f:
        ntu60_labels = [int(line.strip()) for line in f]
    with open(pjoin(DATA_ROOT['NTU'], 'classes.txt'), 'r', encoding='utf-8') as f:
        ntu60_label_texts = [line.strip() for line in f] 
    with open(pjoin(DATA_ROOT['HML'], f'{SPLIT}.txt'), 'r', encoding='utf-8') as f:
        hml3d_samples = [line.strip() for line in f]    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Process NTU data to have only one text per unique label
    if args.use_action_label:
        # Use class label texts directly - one per unique label
        unique_labels = sorted(set(ntu60_labels))
        ntu60_texts = [ntu60_label_texts[label] for label in unique_labels]
        ntu60_labels_unique = unique_labels
        cache_suffix = "action_labels"
    else:
        # Load natural language descriptions and get one per unique label
        print(f"Loading NTU-RGB+D descriptions from {TXT_ROOT['NTU']}...")
        
        # Group samples by label and pick one text per label
        label_to_samples = {}
        for sample, label in zip(ntu60_samples, ntu60_labels):
            if label not in label_to_samples:
                label_to_samples[label] = []
            label_to_samples[label].append(sample)
        
        # Get one random text description per unique label
        unique_labels = sorted(label_to_samples.keys())
        ntu60_texts = []
        
        for label in unique_labels:
            # Pick one random sample for this label
            sample = random.choice(label_to_samples[label])
            file_path = os.path.join(TXT_ROOT['NTU'], f'{sample}.txt')
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip().split('#')[0].strip() for line in f if line.strip()]
                ntu60_texts.append(random.choice(lines))
        
        ntu60_labels_unique = unique_labels
        cache_suffix = "natural_language"

    print(f"Processing {len(ntu60_texts)} unique NTU labels...")
    
    # Load HML3D data
    print(f"Loading HumanML3D descriptions from {TXT_ROOT['HML']}...")
    hml3d_texts = load_texts(TXT_ROOT['HML'], hml3d_samples)

    # Encode embeddings with proper cache file names
    print("Encoding NTU RGB+D texts...")
    ntu60_embeds = get_text_embedding(
        ntu60_texts, model, device, args, 
        use_cache=args.use_cache, 
        out_file=f'ntu60_texts_clip_{cache_suffix}.pt'
    )

    print("Encoding HumanML3D descriptions...")
    hml3d_embeds = get_text_embedding(
        hml3d_texts, model, device, args, 
        use_cache=args.use_cache, 
        out_file='hml3d_texts_clip.pt'
    )

    print("Computing k-NN stats...")
    result, json_file = compute_knn_stats(ntu60_embeds, ntu60_labels_unique, ntu60_texts, hml3d_embeds, hml3d_texts, args, out_path=OUT_PATH)
    
    # Automatically perform ranking after computing stats
    print(f"\nRanking results from: {json_file}")
    df_ranked = rank_ntu_labels_by_crowding(result)