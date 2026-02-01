# LoHRbench: Pi0-FAST Baseline

Training and evaluation code for the **Pi0-FAST** (Physical Intelligence) baseline on LoHRbench, built on top of [OpenPI](https://github.com/Physical-Intelligence/openpi).

GitHub: [https://github.com/HaoranZhangumich/LOHRbench_openpi](https://github.com/HaoranZhangumich/LOHRbench_openpi)

## Repository Structure

```
LOHRbench_openpi/
├── eval_pi0.py                     # LoHRbench evaluation script
├── third_party/
│   └── openpi/                     # OpenPI framework (modified)
│       ├── scripts/
│       │   └── train.py            # Main training entry point
│       ├── src/openpi/
│       │   ├── models/
│       │   │   ├── pi0_fast.py     # Pi0-FAST model config
│       │   │   └── pi0_config.py   # Pi0 base config
│       │   └── training/
│       │       ├── config.py       # Training configs (includes LoHRbench)
│       │       ├── lohrbench_rlds_dataset.py  # RLDS data loader
│       │       └── optimizer.py    # AdamW + cosine decay schedule
│       └── ...
└── README.md
```

## Setup

Follow the OpenPI setup instructions:

```bash
cd third_party/openpi

# Install with uv (recommended)
pip install uv
uv sync

# Or install with pip
pip install -e .
```

## Dataset

Download the demonstration dataset from HuggingFace: **[oldTOM/LoHRbench](https://huggingface.co/datasets/oldTOM/LoHRbench)**

The HuggingFace dataset provides HDF5 files. Pi0 requires **RLDS format**, so you need to convert the HDF5 data first (see below).

## Data Format

Pi0 consumes data in **RLDS format** (TensorFlow Datasets). Convert the downloaded HDF5 trajectories to RLDS using the conversion script in [`TAMPBench/baseline/utils/data_convert.py`](../TAMPBench/baseline/utils/data_convert.py).

Expected RLDS data directory:
```
/data1/LoHRbench_rlds/
└── lohrbench_rlds/
    └── 0.1.0/
        └── ...  (TFRecord files)
```

Each RLDS episode contains:
- `observation/base_rgb`: base camera image (224x224x3, PNG-encoded)
- `observation/hand_rgb`: wrist camera image (224x224x3, PNG-encoded)
- `observation/qpos`: robot joint positions (9-dim)
- `observation/qvel`: robot joint velocities (9-dim)
- `action`: action command (8-dim)
- `language_instruction`: natural language task description

## Training

```bash
cd third_party/openpi

python scripts/train.py pi0_lohrbench_rlds_finetune \
    --exp_name lohrbench_exp \
    --wandb_enabled
```

The training config `pi0_lohrbench_rlds_finetune` is defined in `src/openpi/training/config.py` and contains all hyperparameters.

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Base model | Pi0-FAST |
| Pretrained weights | `pi0_fast_base` |
| PaLI-Gemma variant | `gemma_2b_lora` (LoRA-enabled) |
| Action dimension | 8 (7 joints + 1 gripper) |
| Action horizon | 16 |
| Max token length | 500 |
| Batch size | 32 |
| Optimizer | AdamW (beta1=0.9, beta2=0.95, eps=1e-8, wd=1e-10) |
| Gradient clipping | Global norm = 1.0 |
| LR schedule | Cosine decay |
| &nbsp;&nbsp;Peak LR | 1e-4 |
| &nbsp;&nbsp;Warmup steps | 2,000 |
| &nbsp;&nbsp;Decay steps | 100,000 |
| &nbsp;&nbsp;Final LR | 1e-5 |
| EMA | Disabled (standard for LoRA) |
| Freeze strategy | Freeze all except LoRA layers |
| Save interval | 10,000 steps |
| RLDS shuffle buffer | 250,000 |
| Action chunk size | 16 |
| GPU | 1x NVIDIA A100 80GB |

Checkpoints are saved every 10,000 steps to:
```
checkpoints/pi0_lohrbench_rlds_finetune/<exp_name>/<step>/
```

## Evaluation

### Standalone evaluation (in this repo)

```bash
python eval_pi0.py \
    --checkpoint_dir /path/to/checkpoint/100000 \
    --config_name pi0_lohrbench_rlds_finetune \
    --builder_dir /data1/LoHRbench_rlds/lohrbench_rlds/0.1.0 \
    --num_episodes 5 \
    --max_steps 100
```

### Unified evaluation (via TAMPBench)

```bash
python TAMPBench/baseline/eval.py \
    --policy pi0 \
    --checkpoint /path/to/checkpoint/100000 \
    --config pi0_lohrbench_rlds_finetune \
    --benchmark-root /path/to/TAMPBench/benchmark/table-top \
    --use-action-chunking --chunk-size 16 \
    --results-dir ./results --save-video
```

Set `OPENPI_ROOT` if the evaluation wrapper cannot find OpenPI automatically:

```bash
export OPENPI_ROOT="/path/to/LOHRbench_openpi/third_party/openpi"
```

See the [evaluation README](../TAMPBench/baseline/README.md) for full argument documentation.

## Acknowledgements

Built on top of [OpenPI](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence.