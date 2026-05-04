# Training

<br>

All training commands must be run from within the submodule:

```bash
cd external/motion-diffusion-model
```

Before training, generate a few-shot split — see [other.md](other.md#few-shot-splits).

## 1. Training the Motion Diffusion Model

### MDM

Fine-tune a pre-trained HumanML3D checkpoint on the target dataset:

```bash
python3 -m train.train_mdm \
  --model_type MDM \
  --single_stream target \
  --save_dir ./save/ntu60_mdm \
  --starting_checkpoint ./save/pretraining/humanml_enc_512_50steps/model000750000.pt
```

### KineMIC

Run the full two-stream distillation model:

```bash
python3 -m train.train_mdm \
  --model_type KineMIC \
  --save_dir ./save/ntu60_kinemic \
  --starting_checkpoint ./save/pretraining/humanml_enc_512_50steps/model000750000.pt
```

### Best model (paper configuration)

The command used to produce the best results reported in the paper:

```bash
python3 -m train.train_mdm \
  --model_type KineMIC \
  --save_dir ./save/kinemic_best \
  --starting_checkpoint ./save/pretrain/humanml_enc_512_50steps/model000750000.pt \
  --num_steps 10000 \
  --save_interval 1000 \
  --batch_size 30 \
  --peft LoRA \
  --stream_warmup_steps 0 \
  --lr 2e-5 \
  --grad_clip 1.0 \
  --lambd_rec 1.0 \
  --lambd_window_rec 1.0 \
  --lambd_contrastive 1.0 \
  --lambd_adversarial 0.0 \
  --lambd_window_distill 0.0 \
  --dww
```

<details>
  <summary><b>Argument reference</b></summary>

<br>

**Data**

| Argument | Default | Description |
|---|---|---|
| `--target_dataset` | `ntu-vibe` | `ntu60` \| `ntu120` \| `ntu-vibe` |
| `--target_task_split` | `xsub` | Task split to use |
| `--target_fewshot_id` | `S0000` | ID of the few-shot split to load. Set `NONE` to use the full split |
| `--target_cond_mode` | `mixed` | `action` \| `text` \| `mixed` |
| `--target_data_rep` | `hml_vec` | `hml_vec` \| `xyz` |
| `--prior_checkpoint` | `` | Fixed prior model checkpoint (optional) |

**Model**

| Argument | Default | Description |
|---|---|---|
| `--model_type` | `MDM` | `MDM` (single stream) \| `KineMIC` (dual stream) |
| `--single_stream` | `target` | When using `MDM`, which stream to activate: `prior` \| `target` |
| `--arch` | `trans_enc` | `trans_enc` \| `trans_dec` \| `gru` |
| `--latent_dim` | `512` | Transformer width |
| `--layers` | `8` | Number of transformer layers |
| `--cond_mask_prob` | `0.1` | Classifier-free guidance masking probability |

**Training**

| Argument | Default | Description |
|---|---|---|
| `--save_dir` | required | Output directory for checkpoints |
| `--num_steps` | `5000` | Total training steps |
| `--lr` | `2e-5` | Learning rate |
| `--batch_size` | `64` | Training batch size |
| `--save_interval` | `500` | Checkpoint and eval frequency (steps) |
| `--log_interval` | `25` | Loss logging frequency (steps) |
| `--starting_checkpoint` | `` | Initialize from a pre-trained `.pt` file |
| `--resume_checkpoint` | `` | Resume from a saved checkpoint |
| `--eval_during_training` | flag | Run validation loop during training |
| `--eval_split` | `val` | `val` \| `test` |
| `--gen_during_training` | flag | Generate sample animations at each save interval |
| `--train_platform_type` | `NoPlatform` | `NoPlatform` \| `WandBPlatform` |
| `--use_ema` | flag | Enable EMA model averaging |

**PEFT adapters**

LoRA and MoELoRA can be plugged into the model independently. Avoid assigning the same module to both.

| Argument | Default | Description |
|---|---|---|
| `--peft` | `[]` | Adapters to use: `LoRA` \| `MoELoRA` (or both) |
| `--LoRA_where` | `transformer_attn transformer_ff input_process output_process timestep_embed` | Which modules to apply LoRA to |
| `--LoRA_rank` | `16` | LoRA rank |
| `--LoRA_alpha` | `32` | LoRA alpha |
| `--LoRA_dropout` | `0.1` | LoRA dropout |
| `--MoELoRA_where` | `[]` | Which modules to apply MoELoRA to |
| `--MoELoRA_num_experts` | `6` | Number of MoE experts |
| `--MoELoRA_top_k` | `2` | Top-k expert routing |

**KineMIC loss coefficients**

| Argument | Default | Description |
|---|---|---|
| `--lambd_rec` | `1.0` | Main reconstruction loss weight |
| `--lambd_window_rec` | `1.0` | Prior window reconstruction loss weight |
| `--lambd_window_distill` | `0.5` | Prior window latent distillation loss weight |
| `--lambd_contrastive` | `1.0` | Contrastive loss weight |
| `--lambd_adversarial` | `0.0` | Adversarial loss weight |
| `--judge_type` | `None` | `d` (discriminator) \| `c` (critic) |
| `--judge_reg` | `None` | `sn` (spectral norm) \| `gp` (gradient penalty) |
| `--top_k_sp` | `250` | KNN neighbors for soft-positive mining |
| `--tau` | `0.07` | Contrastive loss temperature |
| `--bank_limit` | `60` | Memory bank size for contrastive loss |
| `--dww` | flag | Dynamic Window Weighting |

> All arguments are defined in [`utils/parser_util.py`](../external/motion-diffusion-model/utils/parser_util.py). Modifying defaults there is recommended when running many experiments.

</details>

For ST-GCN evaluator training, see [other.md](other.md#st-gcn-evaluator).
