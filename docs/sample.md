# Sampling

<br>

All sampling commands must be run from within the submodule:

```bash
cd external/motion-diffusion-model
```

> Model architecture arguments (`--arch`, `--latent_dim`, etc.) are loaded automatically from the checkpoint's `args.json` and should not be overridden manually.

<br>

## 1. Motion Synthesis

Generate motion sequences from a trained model conditioned on action IDs or text prompts.

```bash
python3 -m sample.generate \
  --model_path ./save/ntu60_kinemic/model005000.pt \
  --action_id 99 102 104 \
  --num_repetitions 5
```

Omitting `--action_id` automatically samples from the full set of action classes the model was trained on.

<details>
  <summary><b>Argument reference</b></summary>

<br>

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | Path to the `.pt` checkpoint file |
| `--output_dir` | auto | Output directory. If empty, created next to the checkpoint |
| `--num_samples` | `9` | Number of prompts/actions to sample |
| `--num_repetitions` | `5` | Repetitions per action/prompt |
| `--guidance_param` | `2.5` | Classifier-free guidance scale |
| `--motion_length` | `2.5` | Motion length in seconds |
| `--sampling_stream` | `target` | `prior` \| `target` |
| `--sampling_mode` | `single` | `single` (one stream) \| `cycle` (both streams alternating) |
| `--action_id` | `[]` | One or more action class IDs |
| `--text_prompt` | `` | A single free-form text prompt |
| `--input_text` | `` | Path to a `.txt` file with one text prompt per line |
| `--action_file` | `` | Path to a `.txt` file with one action name per line |
| `--unconstrained_sampling` | flag | Sample without any conditioning |

</details>

<br>

## 2. Build a Synthetic Dataset

Generate a full synthetic dataset mirroring the class and length distribution of the training data. Samples used during training are excluded; the rest is generated synthetically.

```bash
python3 -m sample.synth_dataset_generate \
  --model_path ./save/ntu60_kinemic/model005000.pt \
  --oversample 2.0
```

<details>
  <summary><b>Argument reference</b></summary>

<br>

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | Path to the `.pt` checkpoint file |
| `--oversample` | `2.0` | Scale factor applied to the number of real training samples |
| `--guidance_param` | `2.5` | Classifier-free guidance scale |
| `--sampling_stream` | `target` | Must be `target` for synthetic dataset generation |

</details>

<br>

## 3. Inspect a Synthetic Dataset

Visualize the generated synthetic dataset as a grid of animated skeletons.

```bash
python3 -m sample.synth_dataset_inspect \
  --model_path ./save/ntu60_kinemic/model005000.pt
```

<details>
  <summary><b>Argument reference</b></summary>

<br>

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | Path to the `.pt` checkpoint used to generate the dataset |
| `--num_rows` | `12` | Rows in the visualization grid |
| `--num_cols` | `5` | Columns in the visualization grid |

</details>
