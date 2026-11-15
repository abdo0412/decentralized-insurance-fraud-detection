# Section 5 — Vision Agent (CNN with Transfer Learning & Grad-CAM)

**Author:** Person A (Vision Specialist)  
**Model Version:** `resnet50-v1` (`models/vision_resnet50_v1.pt`)

---

## 5.1 Dataset Preparation & Leak-Free Two-Pool Split

To train the Vision Agent while strictly preventing data leakage into the downstream Decision Agent, we used the **Car Damage Severity Dataset** (`1,631` total verified collision photos across three canonical classes: `minor`, `moderate`, and `severe`).

Before any training took place, `scripts/prepare_vision_data.py` partitioned the raw dataset using stratified sampling (`seed=42`) into two completely disjoint pools:
1. **Held-Out Claims Pool (30% — `490` images):** `160` minor, `162` moderate, and `168` severe images (`data/processed/held_out_images.csv`). The CNN **never** trained or validated on these images. They were reserved exclusively for Person C to attach to tabular insurance claims, ensuring that the Decision Agent receives realistic out-of-sample vision probabilities rather than memorized training scores.
2. **CNN Development Pool (70% — `1,141` images):** Stratified into:
   - **Train (`798` images, 70%):** Used for parameter optimization with on-the-fly data augmentation (`224×224` resize, random horizontal flip $p=0.5$, random rotation $\pm 15^\circ$, color jitter, and ImageNet normalization).
   - **Validation (`171` images, 15%):** Used for learning-rate scheduling and checkpoint selection via validation Macro F1.
   - **Test (`172` images, 15%):** Untouched evaluation split used once at the end to report standalone Vision metrics.

---

## 5.2 Model Architecture & Two-Stage Transfer Learning

We selected **ResNet50** initialized with `IMAGENET1K_V2` pretrained weights (`agents/vision_agent.py`). The original 1,000-class fully connected layer (`fc`) was replaced by `Dropout(p=0.3)` followed by a 3-class linear projection `Linear(2048, 3)`. Class imbalance was addressed via inverse-frequency class weights in the Cross-Entropy loss function.

Training was conducted in two stages on an NVIDIA T4 GPU (`scripts/train_vision_cnn.py`):
- **Stage 1 — Classifier Head Warmup (5 epochs):** All convolutional backbone layers were frozen (`requires_grad=False`), and only the new 3-class head was trained using `AdamW` ($\text{lr} = 10^{-3}$, weight decay $10^{-4}$). This stabilized the randomly initialized head without corrupting pretrained spatial filters.
- **Stage 2 — Deep Block Fine-Tuning (12 epochs):** The higher-level semantic blocks (`layer3` and `layer4`) were unfrozen and fine-tuned jointly with the classification head using `AdamW` ($\text{lr} = 10^{-4}$) and a `CosineAnnealingLR` schedule.

---

## 5.3 Quantitative Results on the Untouched Test Set

On the held-out CNN test set (`172` images), `resnet50-v1` achieved:
- **Overall Test Accuracy:** **71.51%**
- **Macro F1-Score:** **0.7115** (Validation Macro F1: `0.6879`)

| Severity Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Minor** | 0.7460 | 0.8393 | 0.7899 | 56 |
| **Moderate** | 0.5962 | 0.5439 | 0.5688 | 57 |
| **Severe** | 0.7895 | 0.7627 | 0.7759 | 59 |
| **Macro Avg** | **0.7106** | **0.7153** | **0.7115** | **172** |
| **Weighted Avg** | **0.7113** | **0.7151** | **0.7118** | **172** |

### Error Analysis (Confusion Matrix)
As shown in `report/figures/vision_confusion_matrix.png`, the Vision Agent is highly reliable at separating extreme classes:
- **Zero `minor` damage photos were misclassified as `severe`** (`0 / 56`), and only **2 `severe` crashes were misclassified as `minor`** (`2 / 59`).
- This property is critical for our fraud detection objective: the **severity mismatch signal** ($\text{declared\_rank} - E[\text{photo\_rank}]$) specifically targets claimants who declare *Major Damage / Total Loss* (`severe`) while submitting a photo of a light scratch (`minor`). Because the CNN rarely confuses `minor` and `severe`, false-positive severity mismatch flags are kept low.
- Most errors occur along the natural boundary between `moderate` and adjacent classes (`minor` or `severe`), which is expected given subjective visual boundaries in single-angle 2D photos.

---

## 5.4 Visual Explainability via Grad-CAM

To ensure an human auditor can verify *why* a photo received a given severity score, we implemented **Grad-CAM** (`explain/gradcam.py`) hooked into the final convolutional block `model.layer4[-1]`.

Inspection of `report/figures/vision_gradcam_examples.png` confirms that the network's high-activation regions localize directly onto physical vehicle deformation—such as crumpled hoods, detached front bumpers, and dented fenders—rather than background pavement, trees, or neighboring parked cars.

---

## 5.5 Handoff & Limitations

- **Held-Out Pool Scoring:** All `490` held-out photos were scored using `scripts/predict_held_out.py` and exported to `data/processed/vision_predictions.csv` following our standardized `AgentMessage` contract.
- **Limitations:** Single-view exterior photos cannot capture hidden structural or mechanical frame damage, and our synthetic claim-to-photo linking rule is an experimental assumption rather than organic insurer claim data.
