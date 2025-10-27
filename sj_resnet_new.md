# SJ-ResNet Training Notes

This document is written for our Toshiba neuromorphic internship project.  
It explains what the unified training script does, how to launch the key benchmarks (including DVS data), and how to read the logs that matter to Nishi-san’s team.

## 1. Project Goals
- Provide a single entry point (`sj_resnet_full.py`) to train spiking ResNet models on both conventional image datasets and event-based datasets.
- Benchmark “deep-learning style” training quality (accuracy, loss) **and** neuromorphic signals (spike rate, membrane potential histograms) so that the hardware-focused members can follow the experiments.
- Keep experiment outputs reproducible: every run drops TensorBoard logs, config snapshots, and summaries in `./runs/`.

## 2. Pipeline at a Glance
```
          ┌──────────────────┐
          │   YAML Config    │
          │ (model, dataset, │
          │ logging options) │
          └─────────┬────────┘
                    │
        Parse + Create Run Folder
                    │
                    ▼
         ┌────────────────────┐
         │   Training Script  │
         │  (spikingjelly)    │
         └─────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   TensorBoard Logs     Model Checkpoints
 (loss, acc, spikes)         (ckpt.pth)
        │                     │
        └─────────┬──────────┘
                  ▼
      ┌───────────────────────┐
      │ CSV Result Summary    │
      │  (metrics per run)    │
      └─────────┬─────────────┘
                ▼
    ┌───────────────────────┐
    │ Result Dashboard Plot │
    │ (efficiency vs acc)   │
    └───────────────────────┘
```
This is the end-to-end loop we keep iterating on for every experiment.

## 3. Environment Checklist
- Python environment: `/home/is/hanfu-zh/miniconda3/envs/sj_resnet/bin/python`
- Key packages: `torch`, `torchvision`, `spikingjelly`, `tensorboard`, `pyyaml`, `pandas`, `matplotlib`, `seaborn`
- GPU is optional but recommended; script auto-selects CUDA if available.

If TensorBoard complains about missing spikes, ensure you are using `--neuron lif` (membrane potentials are only logged for LIF).

## 4. Quick Start Commands

### 4.1 Baseline (CIFAR-10, static images)
```bash
python sj_resnet_full.py \
  --dataset cifar10 \
  --model spiking_resnet18 \
  --neuron if \
  --T 4 \
  --epochs 5 \
  --tag baseline
```

### 4.2 Event-based Benchmark (CIFAR10-DVS)
First run will download and preprocess ~3.4 GB of data inside `./data/cifar10_dvs`.
```bash
python sj_resnet_full.py \
  --dataset cifar10_dvs \
  --model spiking_resnet18 \
  --neuron lif \
  --T 32 \
  --batch-size 32 \
  --tag dvs
```
- The loader auto-splits the dataset 90%/10% for train/test.
- Resume runs after interruptions: the script will continue any partially downloaded ZIPs.

### 4.3 DVSGesture (hand-gesture events)
```bash
python sj_resnet_full.py \
  --dataset dvsgesture \
  --model spiking_resnet18 \
  --neuron lif \
  --T 32 \
  --batch-size 32 \
  --tag gesture
```
- Dataset saved under `./data/dvsgesture`. As soon as a sample is processed, it becomes reusable—future runs skip re-processing.

### 4.4 YAML one-command run
```bash
python sj_resnet/benchmarks/run_config.py --config configs/cifar10_resnet18_if.yaml
```
- 使用现成 `configs/` YAML。命令会打印日志目录；如需临时覆盖，可追加参数：`-- --epochs 40 --amp`。

### 4.5 汇总与可视化
```bash
python sj_resnet/demo/aggregate_runs.py > aggregated_runs.csv
python sj_resnet/demo/plot_dashboard.py --csv aggregated_runs.csv --output dashboard.png
```
- `aggregated_runs.csv` 记录每次实验的 loss/acc/spike 指标。
- `dashboard.png` 展示 Accuracy ↔ Spikes 散点图，可直接放在汇报或邮件里。

> 想让非工程背景的同事也能上手，可参考 `sj_resnet/tutorial_non_technical.md` 的一步一步操作说明。

## 5. Important Arguments
| Flag | Meaning |
|------|---------|
| `--dataset` | `cifar10`, `cifar100`, `tinyimagenet`, `cifar10_dvs`, `dvsgesture` |
| `--T` | Temporal steps. For frame-based data, frames are repeated `T` times; for DVS data, the loader already returns `T` frames. |
| `--neuron` | `if` (no membrane state) or `lif` (leaky). Choose `lif` for potential histograms. |
| `--amp` | Add for mixed precision (`--amp`). |
| `--save-model` | Stores best accuracy checkpoint in the run directory. |

Full argument list: `python sj_resnet_full.py --help`.

## 6. Outputs and Logging
- **Directory**: `./runs/<experiment_name>` with timestamped folders.
- **Inside each folder**:
  - `config.txt`: JSON dump of CLI arguments.
  - `summary.json`: Final metrics, best accuracy, elapsed time.
  - TensorBoard event files containing:
    - `Train/EpochLoss`, `Train/EpochAcc`, `Test/Loss`, `Test/Acc`
    - `Spikes/Train/<layer>_mean`, `Spikes/Test/<layer>_mean` (average firing rate per layer)
    - `Spikes/Train/Test/GlobalRate` (network-wide rate — sparsity proxy)
    - `PotentialHist/<phase>/<layer>` histograms (only for LIF)
    - `Efficiency/<phase>/SpikesPerSample` (approx total spikes per input across all layers)
    - `PotentialNearThresh/<phase>/<layer>` (fraction of membrane potentials near threshold — efficiency potential)
- **Viewing**: `tensorboard --logdir ./runs` (remember to point colleagues to Spikes and Potential tabs during demos).
- **Aggregated CSV**: `python sj_resnet/demo/aggregate_runs.py > aggregated_runs.csv`
- **Dashboard Plot**: `python sj_resnet/demo/plot_dashboard.py --csv aggregated_runs.csv --output dashboard.png`

## 7. Interpretation Tips for Nishi-san’s Team
- **Spike Scalars**: Values close to 0.05–0.10 indicate sparse activity suitable for low-power hardware. If a layer reports >0.3 consistently, consider reducing `T` or adjusting learning rate.
- **Potential Histograms** (`lif` only): Peak around 0 means neurons are reset; long tails near the threshold show active computation. Negative tails reflect inhibition or lack of excitation—expected in sparse coding.
- **DVS Data**: Each run prints download/extract logs once. After conversion, subsequent experiments start immediately; highlight this when sharing the workflow.

## 8. Recommended Reporting Milestones
1. **CIFAR-10 static baseline** (IF neuron) – Demonstrate parity with mainstream SNN training.
2. **CIFAR10-DVS** (LIF neuron, `T=32`) – Evidence that our pipeline supports neuromorphic event data end-to-end.
3. **Gesture recognition** – Optional, but useful to showcase for real-world appliance scenarios.

For each milestone prepare:
- Command used (ready to copy/paste)
- Final accuracy vs epochs
- Spike/Global rate trend screenshot
- Potential histogram snapshot (for LIF runs)
- Dashboard scatter (Accuracy vs Spikes)

## 9. Troubleshooting
- **Download interrupted**: rerun the same command; already-downloaded archives are reused.
- **No `PotentialHist` charts**: ensure `--neuron lif`.
- **TensorBoard missing spikes entirely**: confirm the run directory; older runs before spike logging upgrade will lack these tags.
- **Import errors for DVS datasets**: script auto-detects module paths, but if the environment changes, reinstall `spikingjelly` (`pip install -U spikingjelly`).

This note should give Nishi-san and the hardware members a clear view of what to run, what to expect, and how to connect the outputs back to their neuromorphic goals.
