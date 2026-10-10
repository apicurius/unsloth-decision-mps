# unsloth-decision-mps

Train an [Unsloth decision model](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)
(LLM + Cloudflare [Clef](https://huggingface.co/Cloudflare/clef) joint schema head, LoRA) on an Apple Silicon GPU (MPS).

Unsloth 2026.10.3 refuses this on Macs: its MLX stub raises
`FastDecisionModel training is not yet supported on MLX`, and the GPU path requires NVIDIA/AMD/Intel.
`mps_shim.py` imports Unsloth's plain-PyTorch decision code (`unsloth/models/decision.py`, `clef.py`,
the vendored Clef `joint_schema_model.py`) without running `unsloth/__init__.py`, and stubs four
GPU-only helpers. The model loads on CPU, then moves to `mps`, and trains in float32.

## Result: Qwen3.5-0.8B on `LocalLLaMA/typed-decisions`

Apple M3 Max (30-core GPU, 36 GB). LoRA r=16/alpha=16, lr 2e-4 (head 1e-4), batch 2 x grad-accum 4,
2 epochs (280 steps), cosine, warmup 10, seed 3407, max_seq_length 2048. Training data: `typed-decisions`
`all` subset only (1,120 train cases; 80 held out for calibration).

| typed-decisions test (400 cases, 2,000 decisions) | Unsloth docs | This repo |
|---|---|---|
| Accuracy before training | 36% | 35.6% |
| Accuracy after training | 73% | **78.4%** |
| ECE after calibration | – | 0.034 |
| All 5 answers right per case | – | 35.3% |
| Training time | 42 min (GPU not stated) | 1 h 49 min (17–27 s/step) |

Held-out (80 cases) accuracy: 80.0%. Per-step loss: `results/train_log.csv` (about 1.16 -> 0.83).
Full metrics: `results/result.json`.

Not a like-for-like comparison: Unsloth's run mixes `typed-decisions` with 12 public datasets and uses
LoRA r=64 for 1 epoch (4-bit). This run is one seed. The test labels come from a ~4B teacher model with
0.735 self-agreement, so scores above that partly measure the teacher's quirks
(see the [dataset card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)).

## Run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
PYTHONUNBUFFERED=1 .venv/bin/python train_decision.py --device mps --model unsloth/Qwen3.5-0.8B \
  --epochs 2 --r 16 --lr 2e-4 --out run
```

`--skip_before` skips the ~9 min untrained test pass. The run writes LoRA adapters and the Clef head
to `run/adapter/` and metrics to `run/result.json`.

## Notes

- 4-bit loading is unavailable (bitsandbytes needs a GPU backend Unsloth supports); training is float32.
- Qwen3.5's linear-attention layers fall back to plain PyTorch (`flash-linear-attention` and
  `causal_conv1d` need CUDA). This is the main reason for the speed gap with NVIDIA GPUs.
- Gradient checkpointing is off on this loading path.
- Unsloth's default loss is cross-entropy; Cloudflare's Clef recipe also uses label smoothing,
  a Brier term and an RL stage.
