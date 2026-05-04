# Other Tools

<br>

## Few-Shot Splits

> **Paper splits:** the exact few-shot splits used in our experiments are already provided at `data/NTU-VIBE/splits/fewshot/S0000`. You only need to run the script below if you want to generate new splits with different seeds, classes, or shot counts.

Generate a random few-shot split from the repository root:

```bash
python3 -m scripts.sample_fewshot_split --dataset NTU60 --seed 42 --shots 10
```

Omitting `--class_list` uses all valid classes (excluding those with multiple skeletons). Splits are produced independently for every task configuration (`xsub`, `xview`, etc.) and include a matching validation split built from the full validation set restricted to the selected classes.

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `NTU60` | `NTU60` \| `NTU120` \| `NTU-VIBE` |
| `--seed` | `42` | Random seed |
| `--shots` | `10` | Samples per class |
| `--class_list` | all | Specific class IDs (e.g. `2 3 19 29`). Omit to use all valid classes |
| `--exclude_outliers` | `True` | Exclude annotated outlier samples |
| `--with-stats` | flag | Compute mean/std for the split |

## ST-GCN Evaluator

The ST-GCN evaluator is used to measure the downstream HAR accuracy of generated synthetic data.

All commands must be run from within the submodule:

```bash
cd external/motion-diffusion-model
```

### On the full dataset or a few-shot split

```bash
# Full dataset
python3 -m train.train_stgcn_evaluator --dataset NTU60 --task_split xsub

# Few-shot split only
python3 -m train.train_stgcn_evaluator --dataset NTU60 --task_split xsub --fewshot_id <split_id>
```

### On synthetic / mixed data

```bash
python3 -m train.train_har \
  --data_dir ./datasets/NTU60/splits/xsub/S0001 \
  --mode mixed \
  --data_rep j
```

`--mode` controls the training set composition: `synth` (synthetic only) · `real` (few-shot real only) · `mixed` (both) · `gt` (all real samples from few-shot classes).

<details>
  <summary><b>Argument reference</b></summary>

<br>

**train_stgcn_evaluator**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `NTU60` | `NTU60` \| `NTU120` \| `NTU-VIBE` |
| `--task_split` | `xsub` | `xsub` \| `xview` |
| `--fewshot_id` | `None` | Few-shot split ID. Omit to train on the full dataset |
| `--num_epochs` | `80` | Training epochs |
| `--batch_size` | `64` | Batch size |
| `--seed` | `42` | Random seed |
| `--overwrite` | flag | Overwrite existing save directory |
| `--train_platform_type` | `NoPlatform` | `NoPlatform` \| `WandBPlatform` |

**train_har**

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Root folder of the data split (real, synthetic, or mixed) |
| `--mode` | `mixed` | `synth` \| `real` \| `mixed` \| `gt` |
| `--data_rep` | `j` | `j` (joints) \| `b` (bones) \| `jm` \| `bm` |
| `--incremental` | `-1` | Max real samples to include. `-1` uses all available |
| `--seed` | `1 2 3 4 5` | One or more seeds (runs once per seed) |
| `--use_wandb` | flag | Enable W&B logging |

</details>
