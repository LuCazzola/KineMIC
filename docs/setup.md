# Setup

<br>

## 1. Clone the Repository

```bash
git clone --recursive https://github.com/LuCazzola/KineMIC.git
cd KineMIC
git submodule update --remote
```

<br>

## 2. Dependencies

Install system and Python dependencies:

```bash
sudo apt update && sudo apt install ffmpeg
```

```bash
conda env create -f environment.yml
conda activate cloudspace
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Download the following resource archives and place them under `./external/motion-diffusion-model`:

| File | Description |
|---|---|
| [smpl.zip](https://drive.usercontent.google.com/download?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2&authuser=1) | SMPL body model |
| [glove.zip](https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing) | GloVe word embeddings |
| [t2m.zip](https://drive.usercontent.google.com/download?id=1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb&authuser=1) | HumanML3D evaluation models |
| [kit.zip](https://drive.usercontent.google.com/download?id=12liZW5iyvoybXD8eOw4VanTgsMtynCuU&authuser=1) | KIT evaluation models |

Then unpack everything:

```bash
bash prep/init_dep.sh
```

<br>

## 3. Data

Both datasets used in this work are **licensed** and must be requested directly from their respective sources.

### HumanML3D (prior domain)

Follow the instructions in the [official HumanML3D repository](https://github.com/EricGuo5513/HumanML3D) to obtain and generate the dataset, then place it at:

```
./external/motion-diffusion-model/dataset/HumanML3D
```

### NTU RGB+D (target domain)

NTU RGB+D is available upon request through the [ROSE Lab dataset portal](https://rose1.ntu.edu.sg/dataset/actionRecognition/).

Once access is granted, re-estimate SMPL skeletons from the RGB videos using [VIBE](https://github.com/mkocabas/VIBE) or another SMPL-based pose estimator. In our paper we use VIBE.

> **Our pre-estimated skeletons:** we may share our VIBE-estimated skeletons upon request, but only after you have obtained valid licenses for both NTU RGB+D and HumanML3D. Contact us with proof of access.

After estimation, organize the raw SMPL output under:

```
./data/NTU-VIBE-RAW/<subject_id>/<sample_name>.npy
```

Then run the pre-processing script to convert to MDM-compatible format:

```bash
python3 -m scripts.ntu_vibe_preproc
```

<details>
  <summary><b>Argument reference</b></summary>

<br>

| Argument | Default | Description |
|---|---|---|
| `--in_dataset` | `NTU-VIBE-RAW` | Input directory name under `./data/` containing raw VIBE skeletons |
| `--out_dataset` | `NTU-VIBE` | Output dataset name |
| `--label_filter` | `99 102 104` | Action class IDs to process. Omit to process all valid classes |
| `--skip_skel_preproc` | flag | Skip skeleton conversion (useful to re-run only text formatting) |

</details>

Finally, symlink the processed data into the submodule:

```bash
bash prep/link_data.sh NTU-VIBE
```

<br>

## 4. Pre-Trained MDM Checkpoint

Download the pre-trained MDM checkpoint and extract it under `./external/motion-diffusion-model`:

| File | Description |
|---|---|
| [humanml_enc_512_50steps.zip](https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view?usp=sharing) | MDM encoder, 512-dim, 50 diffusion steps |

```bash
cd external/motion-diffusion-model
mkdir -p save/pretrain
unzip humanml_enc_512_50steps.zip -d save/pretrain/
rm humanml_enc_512_50steps.zip
```

> This is an updated checkpoint with comparable performance to the one in the original MDM paper but **×20 faster** inference. It is used as the transformer encoder-based prior in KineMIC.
