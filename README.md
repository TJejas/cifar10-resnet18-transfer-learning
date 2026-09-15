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
- **Data:** CIFAR-10 (50k train / 10k test), resized to 224x224 to match the
  ImageNet-pretrained input distribution, normalized with ImageNet mean/std.
  Train-time augmentation: random horizontal flip + random crop.
- **Training:** Adam optimizer, cosine annealing LR schedule, mixed-precision
  (`torch.autocast` + `GradScaler`) on GPU.

## Results

See [`outputs/results.json`](outputs/results.json) and
[`outputs/training_curves.png`](outputs/training_curves.png) for the actual
run.

| Metric | Value |
|---|---|
| Best test accuracy | _filled in after training_ |
| Epochs | _filled in after training_ |
| Hardware | _filled in after training_ |

## Usage

```bash
pip install -r requirements.txt
python train.py --epochs 10 --batch-size 32
```

CIFAR-10 downloads automatically into `data/` on first run. The best
checkpoint (by test accuracy) is saved to `checkpoints/best_model.pth`
(not tracked in git - re-run `train.py` to reproduce it).

Useful flags:

```bash
python train.py --subset 2000 --epochs 1   # quick smoke test
python train.py --img-size 160             # smaller/faster
```
