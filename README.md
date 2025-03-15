# Smoothness-aware Little Robust Training

This is the code implementation of the Smoothness-aware Little Robust Training method.

## Installation

Set up the code environment using:
```bash
pip install -r requirements.txt
```

---

## Training & Evaluation Workflow

### Imagenette Dataset

1. **Train a Baseline Model** (e.g., ResNet50):
```bash
python main_asam_imagenette_v2.py \
    --saved_dir ./imagenette_logs/check_points_v2_torchvision/ \
    --train_type 7 \
    --random_seeds 1 \
    --epoch 100 \
    --arch 'resnet50' \
    --batch_size 64
```

2. **Implement Smoothness-aware Little Robust Training**:
```bash
python main_asam_imagenette_v2.py \
    --saved_dir ./imagenette_logs/check_points/ \
    --train_type 3 \
    --loss_schema averaged \
    --fuse_weight 0.2 \
    --random_seeds 1 \
    --epoch 100 \
    --attack_lr 0.4 \
    --eps 1 \
    --rho 0.6 \
    --batch_size 48
```

3. **Evaluate Results**:
```bash
python eval.py \
    --saved_excel_path ./logs/saved_result \
    --method_name your_method_name \
    --dataset imagenette
```

---

### CIFAR10 Dataset

Follow the same workflow as Imagenette, but use different command to train the baseline model:

**Baseline Model Training** (using Robustness package):
```bash
python -m robustness.main \
    --dataset cifar \
    --data ./path_to_cifar10 \
    --adv-train 0 \
    --arch resnet18 \
    --out-dir ./logs/resnet18
```

---

## Model Smoothness Evaluation

Assess model smoothness using:
```bash
python compute_hessian.py \
    --mini-hessian-batch-size 100 \
    --hessian-batch-size 10000 \
    --data-path ./path_to_cifar10 \
    --saved_excel_path ./saved_result/smoothness/random_seeds_1 \
    --method_name your_method_name
```

---

### Key Notes:
- Replace `your_method_name` with your actual method identifier
- Ensure all file paths match your local directory structure
- Adjust batch sizes according to your GPU memory constraints
