"""Builds notebooks/breast_cancer_model_comparison.ipynb — the executable
research notebook for the 9-model IDC histopathology comparison. Cells call
directly into scripts/train_idc_comparison.py's tested functions (dataset,
model builders, run_model) rather than duplicating logic, per the project's
"reusable functions, but the notebook must call them and show real output"
requirement.

    python -m scripts.build_comparison_notebook

Building the notebook is separate from executing it — see scripts/run_notebook.py.
"""
import nbformat as nbf
from scripts.train_idc_comparison import MODEL_SPECS

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------- Title
md("""# Breast Cancer Histopathology — CNN vs Transformer vs Hybrid Model Comparison

**Dataset:** IDC (Invasive Ductal Carcinoma) histopathology patches, a 50-patient / 48,759-patch
subset of the 279-patient / 277,524-patch Breast Histopathology Images dataset (Janowczyk &
Madabhushi), extracted from a user-supplied `IMAGE.zip`.

**Task:** binary patch-level classification — non-IDC (class 0) vs IDC (class 1).

**Research question:** does model performance actually improve as architecture complexity
increases — from a from-scratch CNN, through CNN transfer learning, to Vision Transformers,
to a CNN+Transformer hybrid — on this dataset and hardware, or not? The answer is determined
by the actual results below, not assumed in advance.

**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM), PyTorch 2.6.0+cu124, CUDA 12.4.

**Important limitation, stated upfront:** this uses 50 of 279 patients (~18% of the full
dataset) for practicality on a 4GB-VRAM laptop GPU within a limited timeframe. Results here
demonstrate architecture comparison trends on this subset — they are not claimed to represent
the full dataset's population statistics, and this is a research/academic exercise, not a
clinically validated result.""")

# ---------------------------------------------------------------- Section 1: setup
md("## 1. Environment Setup and GPU Verification")

code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import json
import torch
import pandas as pd
from IPython.display import Image as IPyImage, display

from scripts.train_idc_comparison import (
    DATA_DIR, SPLIT_PATH, MODELS_DIR, RESULTS_DIR, COMPARISON_CSV, DEVICE,
    disk_free_gb, gpu_status, IDCDataset, MODEL_SPECS, ARCH_TYPE, run_model,
)

print('Python:', sys.version.split()[0])
print('PyTorch version:', torch.__version__)
print('torchvision version:', __import__('torchvision').__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('Device selected:', DEVICE)
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f'GPU VRAM: {free/1e9:.2f} GB free / {total/1e9:.2f} GB total')
print('Disk free:', f'{disk_free_gb():.2f} GB')""")

# ---------------------------------------------------------------- Section 2: dataset
md("""## 2. Dataset Loading

Loading the verified 50-patient IDC subset from `data/raw/histopathology_idc/`. Each patient
has two subfolders: `0/` (non-IDC patches) and `1/` (IDC patches), each a 50x50 RGB PNG.""")

code("""print('Dataset path:', DATA_DIR)
patient_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
print(f'Patient folders found: {len(patient_dirs)}')

class0_total = class1_total = 0
for p in patient_dirs:
    class0_total += len(list((p / '0').glob('*.png'))) if (p/'0').exists() else 0
    class1_total += len(list((p / '1').glob('*.png'))) if (p/'1').exists() else 0
total = class0_total + class1_total
print(f'Total patches: {total}')
print(f'Class 0 (non-IDC): {class0_total} ({class0_total/total*100:.1f}%)')
print(f'Class 1 (IDC):     {class1_total} ({class1_total/total*100:.1f}%)')
print()
print('Class 0 = benign/normal tissue patch (no invasive ductal carcinoma present)')
print('Class 1 = invasive ductal carcinoma (IDC) present in this patch')""")

code("""# Display sample images from each class
import matplotlib.pyplot as plt
from PIL import Image
import random

rng = random.Random(42)
fig, axes = plt.subplots(2, 6, figsize=(15, 5))
for col in range(6):
    p = rng.choice(patient_dirs)
    for row, cls in enumerate(('0', '1')):
        files = list((p / cls).glob('*.png'))
        if not files:
            continue
        f = rng.choice(files)
        img = Image.open(f)
        axes[row][col].imshow(img)
        axes[row][col].set_title(f'patient {p.name}\\nclass {cls}', fontsize=9)
        axes[row][col].axis('off')
fig.suptitle('Sample IDC patches (top: class 0 non-IDC, bottom: class 1 IDC)')
fig.tight_layout()
plt.show()""")

# ---------------------------------------------------------------- Section 3: split
md("""## 3. Patient-Level Train/Validation/Test Split

Loading `data/splits/idc_patient_split.json` — the SAME split used for every model below.
Split is at the **patient** level: no patient's patches appear in more than one of
train/validation/test, preventing leakage.""")

code("""with open(SPLIT_PATH) as f:
    manifest = json.load(f)

split = manifest['split']
print(f\"Train patients: {len(split['train'])}\")
print(f\"Val patients:   {len(split['val'])}\")
print(f\"Test patients:  {len(split['test'])}\")
print()

for name in ('train', 'val', 'test'):
    s = manifest['split_summary'][name]
    print(f\"{name:5s}: {s['patients']:2d} patients, class0={s['class0']:6d}, class1={s['class1']:6d}, total={s['total']:6d}\")

print()
tv = set(split['train']) & set(split['val'])
tt = set(split['train']) & set(split['test'])
vt = set(split['val']) & set(split['test'])
print('LEAKAGE CHECK')
print('Patient overlap train/val:', len(tv))
print('Patient overlap train/test:', len(tt))
print('Patient overlap val/test:', len(vt))
assert not tv and not tt and not vt, 'LEAKAGE DETECTED'
print('=> Zero patient overlap confirmed.')""")

# ---------------------------------------------------------------- Section 4: preprocessing
md("""## 4. Preprocessing and Augmentation

Same preprocessing policy applied consistently for every model: ImageNet mean/std
normalization (required for transfer learning with ImageNet-pretrained weights), resized to
each architecture's expected input size. Custom CNN (trained from scratch, no pretrained
weights) uses a smaller 64x64 input since it doesn't need ImageNet-scale resolution.
Augmentation (horizontal flip, small rotation, mild color jitter) is applied to the TRAINING
set only — validation and test sets are never augmented.""")

code("""sample_train_ds = IDCDataset(split['train'][:1], img_size=224, augment=True)
sample_eval_ds = IDCDataset(split['val'][:1], img_size=224, augment=False)
print('Train transform (augmented):')
print(sample_train_ds.transform)
print()
print('Val/Test transform (no augmentation):')
print(sample_eval_ds.transform)""")

# ---------------------------------------------------------------- Section 5: models
md("""## 5. Model Experiments

Training all 9 models **sequentially** (never simultaneously), each on the identical
train/val/test patient split above. Every model:

- Runs a sanity check first (forward + backward + optimizer step on 2 samples, confirms CUDA)
- Trains with mixed precision (`torch.autocast`) on the RTX 3050
- Uses a class-weighted loss (the dataset is ~68%/32% imbalanced) — pos_weight computed from
  the actual training-set class counts
- Pretrained models use two-stage transfer learning: Stage 1 freezes the backbone and trains
  only the classification head (3 epochs), Stage 2 unfreezes everything and fine-tunes at a
  lower learning rate (up to 5 epochs) with early stopping (patience=3, monitored on
  validation ROC-AUC)
- Saves **only the best checkpoint** (by validation ROC-AUC — never chosen using the test set)
- Is evaluated once, at the end, on the held-out test set
- Appends its row to `results/comparison/final_image_model_comparison.csv` immediately, so
  a crash partway through does not lose already-completed results

Pos_weight for the loss is computed from the actual train-split class counts (not assumed).""")

code("""c0 = manifest['split_summary']['train']['class0']
c1 = manifest['split_summary']['train']['class1']
pos_weight_val = c0 / max(c1, 1)
print(f'Train class0={c0} class1={c1} -> pos_weight={pos_weight_val:.4f}')""")

MODEL_DESCRIPTIONS = {
    "custom_cnn": (
        "Custom CNN",
        "From-scratch 4-block CNN (Conv-BN-ReLU x2 + MaxPool per block, 32->64->128->256 channels), "
        "no pretrained weights. This is the baseline: what a plain deep-learning approach achieves "
        "with zero transfer learning, before any of the more advanced architectures enter the comparison.",
        "64x64 input, from-scratch initialization, Adam, up to 10 epochs, early stopping patience=3.",
    ),
    "resnet50": ("ResNet50", "Classic residual CNN, ImageNet-pretrained (timm `resnet50`). The most widely used CNN transfer-learning baseline in medical imaging literature.", "224x224 input, two-stage transfer learning (freeze->fine-tune)."),
    "densenet121": ("DenseNet121", "Densely-connected CNN, ImageNet-pretrained (timm `densenet121`). Dense connections encourage feature reuse — often a strong small-data medical-imaging baseline.", "224x224 input, two-stage transfer learning."),
    "efficientnetv2_s": ("EfficientNetV2-S", "Compound-scaled CNN, ImageNet-pretrained (timm `tf_efficientnetv2_s`). Designed for a strong accuracy/compute tradeoff.", "224x224 input, two-stage transfer learning."),
    "convnext_tiny": ("ConvNeXt-Tiny", "A modernized, transformer-influenced pure-CNN architecture, ImageNet-pretrained (timm `convnext_tiny`). Represents how far CNN design can be pushed before actually adopting attention.", "224x224 input, two-stage transfer learning."),
    "vit_b16": ("ViT-B/16", "Vision Transformer (Base, 16x16 patches), ImageNet-pretrained (timm `vit_base_patch16_224`). No convolutional inductive bias at all — pure self-attention over image patches.", "224x224 input, two-stage transfer learning."),
    "deit_s": ("DeiT-S", "Data-efficient image Transformer (Small), ImageNet-pretrained (timm `deit_small_patch16_224`). Designed specifically to need less pretraining data than plain ViT.", "224x224 input, two-stage transfer learning."),
    "swin_tiny": ("Swin-Tiny", "Hierarchical Shifted-Window Transformer, ImageNet-pretrained (timm `swin_tiny_patch4_window7_224`). Combines transformer attention with CNN-like multi-scale hierarchy.", "224x224 input, two-stage transfer learning."),
    "hybrid_coatnet": ("Hybrid CNN+Transformer (CoAtNet)", "CoAtNet (timm `coatnet_1_rw_224`) — early layers are convolutional, later layers are transformer attention blocks. Used as the CNN+Transformer hybrid entry since no literal 'CNN+Swin' model exists in timm; this is the closest published, real hybrid architecture.", "224x224 input, two-stage transfer learning."),
}

for name, timm_name, img_size, pretrained in MODEL_SPECS:
    title, arch_desc, train_desc = MODEL_DESCRIPTIONS[name]
    md(f"""### 5.{['custom_cnn','resnet50','densenet121','efficientnetv2_s','convnext_tiny','vit_b16','deit_s','swin_tiny','hybrid_coatnet'].index(name)+1} {title}

**Objective:** train and evaluate {title} on the identical 50-patient train/val/test split, using the same test set as every other model for a fair comparison.

**Architecture:** {arch_desc}

**Training setup:** {train_desc}""")
    code(f"""result_status = run_model('{name}', {repr(timm_name)}, {img_size}, {pretrained}, split, pos_weight_val)""")
    code(f"""# Evaluation plots for {title} (only shown if the model actually completed)
_out_dir = RESULTS_DIR / '{name}'
if (_out_dir / 'metrics.json').exists():
    with open(_out_dir / 'metrics.json') as f:
        _m = json.load(f)
    print(json.dumps(_m['test_metrics'], indent=2))
    for _plot in ['training_curves.png', 'roc_curve.png', 'pr_curve.png', 'confusion_matrix.png']:
        _p = _out_dir / _plot
        if _p.exists():
            display(IPyImage(filename=str(_p)))
else:
    print('{title}: no metrics.json found — model did not complete successfully (see STATUS above).')""")

# ---------------------------------------------------------------- Section 6: final comparison
md("""## 6. Final Model Comparison

Reading back `results/comparison/final_image_model_comparison.csv` — built incrementally as
each model above finished, so this reflects only models that actually completed (or explicitly
records failures/skips, never fabricated values).""")

code("""df = pd.read_csv(COMPARISON_CSV)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 140)
display(df)""")

code("""import matplotlib.pyplot as plt

ok = df[df['status'] == 'SUCCESS'].copy()
if len(ok) == 0:
    print('No models completed successfully — nothing to plot.')
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc', 'pr_auc']
    ok.set_index('model')[metrics_to_plot].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Metric comparison across models')
    axes[0].set_ylabel('score')
    axes[0].legend(loc='lower right', fontsize=8)
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(ok['model'], ok['roc_auc'], color='steelblue')
    axes[1].set_title('ROC-AUC by model (primary medical-relevance metric)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    plt.savefig(RESULTS_DIR.parent / 'comparison' / 'model_comparison.png', dpi=100)
    plt.show()""")

# ---------------------------------------------------------------- Section 7: interpretation
md("## 7. Data-Driven Summary")

code("""ok = df[df['status'] == 'SUCCESS']
failed = df[df['status'] != 'SUCCESS']

print('MODELS COMPLETED SUCCESSFULLY:', len(ok), '/', len(df))
if len(failed):
    print()
    print('MODELS NOT COMPLETED:')
    for _, row in failed.iterrows():
        print(f\"  {row['model']}: {row['status']}\")

if len(ok):
    print()
    best_acc = ok.loc[ok['accuracy'].idxmax()]
    best_recall = ok.loc[ok['recall'].idxmax()]
    best_auc = ok.loc[ok['roc_auc'].idxmax()]
    best_f1 = ok.loc[ok['f1'].idxmax()]
    print(f\"Best Accuracy:    {best_acc['model']} ({best_acc['accuracy']:.4f})\")
    print(f\"Best Recall:      {best_recall['model']} ({best_recall['recall']:.4f})\")
    print(f\"Best ROC-AUC:     {best_auc['model']} ({best_auc['roc_auc']:.4f})\")
    print(f\"Best F1:          {best_f1['model']} ({best_f1['f1']:.4f})\")

    cnn_scratch = ok[ok['architecture_type'] == 'CNN (from scratch)']
    cnn_transfer = ok[ok['architecture_type'] == 'CNN (transfer)']
    transformer = ok[ok['architecture_type'] == 'Transformer']
    hybrid = ok[ok['architecture_type'] == 'Hybrid CNN+Transformer']
    print()
    print('Architecture-family averages (ROC-AUC):')
    for label, grp in [('CNN from scratch', cnn_scratch), ('CNN transfer learning', cnn_transfer), ('Transformer', transformer), ('Hybrid CNN+Transformer', hybrid)]:
        if len(grp):
            print(f'  {label}: mean ROC-AUC = {grp[\"roc_auc\"].mean():.4f} (n={len(grp)})')
        else:
            print(f'  {label}: no completed models')""")

md("""## 8. Discussion

*(Read directly off the printed summary above — this section does not restate numbers that
could drift from the actual run; re-run the notebook and the text below still applies
structurally even if the specific winning model changes.)*

**Why accuracy alone is not sufficient here:** the dataset is imbalanced (~68% non-IDC / 32%
IDC in this subset). A model that always predicts "non-IDC" scores ~68% accuracy while
catching zero actual cancer patches (recall = 0). That is why Recall/Sensitivity, Specificity,
ROC-AUC, and PR-AUC are reported for every model, not just accuracy — a high-accuracy,
low-recall model would be a **worse** clinical-support tool, not a better one, because it
would systematically miss malignant patches (false negatives), which is the costliest error
type in cancer screening.

**Whether more complex architectures actually help:** compare the "Architecture-family
averages" printed above. Do NOT assume Transformers or the Hybrid model beat CNNs by default —
transformer architectures generally need substantially more training data than a 50-patient
(48,759-patch) subset provides to outperform CNNs' built-in convolutional inductive bias. If
the printed averages show CNN transfer-learning models matching or beating the
Transformer/Hybrid entries, that is a real, literature-consistent finding for this
data-constrained regime — not a failure of the experiment.

**Limitations of this experiment:**
- Only 50 of 279 patients (~18%) were used, for practicality on a 4GB-VRAM laptop GPU within a
  limited timeframe — results describe trends on this subset, not the full population.
- Validation and test sets are only 8 and 7 patients respectively — small enough that their
  exact class balance depends on which patients the fixed seed happened to select, adding
  variance to the reported metrics.
- No hyperparameter search was performed beyond the fixed two-stage transfer-learning recipe
  described above; a more extensive sweep might shift relative rankings between architectures.
- This is an academic/research comparison. No claim of clinical readiness or clinical
  superiority is made for any model here.""")

nb["cells"] = cells

import pathlib
out_path = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "breast_cancer_model_comparison.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote", out_path, "with", len(cells), "cells")
