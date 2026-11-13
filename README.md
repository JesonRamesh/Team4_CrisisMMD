# Team 4 – CrisisMMD Multimodal Disaster Classification

Shared environment for the CrisisMMD coursework (Parts A, B and C). It provides data loading, the image-feature cache, a text vocabulary and environmental run logging. Models, training and evaluation are implemented by each owner on their feature branch.

## Repository layout

```
.
├── data/CrisisMMD_v2.0/     Raw dataset (git-ignored)
├── features/                Cached ResNet18 features (git-ignored)
├── checkpoints/             Model weights (git-ignored)
├── logs/                    Training logs (git-ignored)
├── results/                 Metrics, predictions (committed)
├── figures/                 Plots (committed)
├── src/
│   ├── config.py            Paths, labels, brief-defined hyperparameters, energy assumptions
│   ├── data.py              Split loading (IDs as strings), label encoding, image-feature alignment
│   ├── text.py              Tokeniser and training-only vocabulary (64-token limit)
│   ├── utils.py             Seeding, device selection, parameter counting
│   ├── env_logger.py        Environmental run logger
│   ├── cache_features.py    Frozen ResNet18 feature extraction
│   └── models/              Model code (one module per owner)
├── env_tracker.csv          Environmental log: one row per run
└── requirements.txt
```

## Setup

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
export CARBON_INTENSITY_G_PER_KWH=207.05   # default: UK DESNZ 2024 annual average
```

## Dataset

Everyone downloads CrisisMMD v2.0 from the official Hugging Face repository, [QCRI/CrisisMMD](https://huggingface.co/datasets/QCRI/CrisisMMD) (about 1.9 GB, CC BY-NC-SA 4.0):

```bash
hf download QCRI/CrisisMMD --repo-type dataset --local-dir data/CrisisMMD_v2.0
```

The brief requires the official **agreed-label** splits (Part A 9,601 / 1,573 / 1,534; Part B 6,126 / 998 / 955). The Hugging Face JSON splits are the all-label splits (13,608 / 2,237 / 2,237) and must not be used for Parts A and B. Add the agreed-label TSVs from the official CrisisMMD data-split release (brief reference [1]) so the layout is:

```
data/CrisisMMD_v2.0/
├── data_image/                           from Hugging Face
├── crisismmd_datasplit_agreed_label/     task_{informative,humanitarian}_text_img_agreed_lab_{train,dev,test}.tsv
└── ...                                   remaining Hugging Face files
```

Image paths in the TSVs (`data_image/<event>/<date>/<id>.jpg`) match the Hugging Face layout. Set `CRISISMMD_ROOT` if the data lives elsewhere.

## Shared image-feature cache

Do not regenerate the cache. Download the shared file from the team drive into `features/` and verify it:

```bash
shasum -a 256 features/resnet18_imagenet1k_v1.pt
# c8443e3e840cb487bfe121ab3e7ce3f88934eb2c3ad643f4c9705cc000baeb68
chmod 444 features/resnet18_imagenet1k_v1.pt   # read-only
```

| Property | Value |
|---|---|
| Images | 18,082 (every image in CrisisMMD v2.0; covers all Part A, B and C splits) |
| Features | `float32 [18082, 512]`, ResNet18 `IMAGENET1K_V1`, eval mode, torchvision transforms |
| Invalid images | 0 |
| Size | 37.6 MB |
| Logged as | `FEAT-resnet18-all-final-01` in `env_tracker.csv` |

The file is a dict with `image_id` (list of strings), `features`, `invalid` and `weights`; load it with `src.data.load_image_features()`. It was generated with:

```bash
python -m src.cache_features --all-images --num-workers 4 --exp-id FEAT-resnet18-all-final-01
```

## Using the shared modules

```python
from src.data import load_split, encode_labels, load_image_features, align_image_features
from src.text import Vocab

train = load_split("informative", "train")
vocab = Vocab.build(train["tweet_text"])                    # training split only
x_text = vocab.encode_batch(train["tweet_text"])            # [N, 64]
y = encode_labels(train, "informative")
feats, index, invalid = load_image_features()
x_img, has_img = align_image_features(train, feats, index)  # [N, 512], [N]
```

## Environmental protocol

> **No code is merged into `main` unless every run behind it (debug, exploratory, final and failed) is logged in `env_tracker.csv` with runtime and energy.** Never delete rows.

Wrap each run in `EnvRun`. It measures energy with CodeCarbon (falling back to the guidance §5.2 estimate), computes emissions, and appends a row on exit, including when the run crashes or is interrupted.

```python
from src.env_logger import EnvRun

with EnvRun("A-inf-textcnn-s0-dbg-01", part="A", task="informative", run_type="debug",
            model="textcnn", seed=0, batch_size=64) as run:
    ...  # training
    run.update(examples_processed=9601, epochs=12, val_macro_f1=0.78,
               run_status="early_stopped")
```

For runs outside Python code (e.g. a notebook cell), log manually:

```bash
python -m src.env_logger --experiment-id A-inf-textcnn-s0-dbg-02 --part A --task informative \
  --run-type debug --model textcnn --runtime-s 84 --run-status completed
```

- **Columns:** experiment ID, part, task, run type, model, seed, examples processed, epochs, batch size, device, runtime (s), energy (kWh), carbon intensity (gCO₂e/kWh), emissions (kgCO₂e), validation macro-F1, run status, cache/checkpoint used.
- **Experiment ID:** `<part>-<task>-<model>-s<seed>-<dbg|exp|final>-<nn>`. Must be unique.
- **run_type:** `debug`, `exploratory`, `final`, `failed`
- **run_status:** `completed`, `early_stopped`, `failed`, `interrupted`

## Git workflow

Feature-branch workflow. Each member works on their own branch and merges it into `main` when it is ready.

| Branch | Scope |
|---|---|
| `feature/part-a-baseline` | Majority-class baseline and metrics |
| `feature/part-a-textcnn` | Text-only models |
| `feature/part-a-image-only` | Image-only model on cached features |
| `feature/part-a-fusion` | Fusion model and TextCNN/BiLSTM swap |

Parts B and C follow the same pattern (`feature/part-b-<scope>`, `feature/part-c-<scope>`), with branches agreed when each phase starts.

```bash
git checkout main && git pull
git checkout -b feature/part-a-textcnn

git add src/ env_tracker.csv
git commit -m "Add TextCNN text-only model"

git push -u origin feature/part-a-textcnn

# merge into main when ready
git checkout main && git pull
git merge feature/part-a-textcnn
git push origin main
```

Merge locally with Git, not through the GitHub web interface: `env_tracker.csv` uses Git's union merge driver, so concurrent log rows combine automatically during a local merge.

**Before merging into `main`:**
- Every run from the branch is in `env_tracker.csv`.
- Vocabulary and any class weights are fitted on training data only.
- No data, feature caches or checkpoints are committed.
