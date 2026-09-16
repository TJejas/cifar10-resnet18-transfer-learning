# CIFAR-10 Image Classification with Transfer Learning (ResNet-18)

A transfer-learning image classifier built with PyTorch. An ImageNet-pretrained
ResNet-18 is fine-tuned on CIFAR-10 by freezing all layers except the final
residual block (`layer4`) and a newly-initialized fully connected head.

This was my first hands-on machine learning project outside coursework.

## Approach

- **Backbone:** `torchvision.models.resnet18` with `IMAGENET1K_V1` weights.
- **Fine-tuning strategy:** all layers frozen except `layer4` and the
  replaced `fc` head (10-way linear classifier), trained with separate
  learning rates (backbone `1e-4`, head `1e-3`).
- **Data:** CIFAR-10 (50k train / 10k test), resized up from 32x32 (160x160 in
  the actual run below, configurable up to 224x224) to better match the
  ImageNet-pretrained input distribution, normalized with ImageNet mean/std.
  Train-time augmentation: random horizontal flip + random crop.
- **Training:** Adam optimizer, cosine annealing LR schedule, mixed-precision
  (`torch.autocast` + `GradScaler`) on GPU.

## Results

See [`outputs/results.json`](outputs/results.json) and
[`outputs/training_curves.png`](outputs/training_curves.png) for the full
run.

| Metric | Value |
|---|---|
| Best test accuracy | **93.88%** (epoch 10) |
| Epochs | 10 |
| Batch size | 16 |
| Image size | 160x160 |
| Hardware | NVIDIA GeForce MX230 (2GB VRAM, laptop GPU) |
| Total training time | ~8.1 hours |

Test accuracy climbed steadily every epoch (90.2% -> 93.9%) with train accuracy
reaching 98.2% by the end — a healthy train/test gap with no significant
overfitting. Full per-epoch numbers are in
[`outputs/results.json`](outputs/results.json).

## Usage

```bash
pip install -r requirements.txt
python train.py --epochs 10 --batch-size 16 --img-size 160
```

Batch size and image resolution were reduced from the defaults (32 / 224) to
fit training on a 2GB-VRAM GPU alongside an 8GB-RAM system. Pass `--resume` to
continue from `checkpoints/last.pth` if a run is interrupted — progress is
checkpointed after every epoch.

CIFAR-10 downloads automatically into `data/` on first run. The best
checkpoint (by test accuracy) is saved to `checkpoints/best_model.pth`
(not tracked in git - re-run `train.py` to reproduce it).

Useful flags:

```bash
python train.py --subset 2000 --epochs 1        # quick smoke test
python train.py --img-size 224 --batch-size 32  # higher resolution (needs more VRAM/RAM)
```
