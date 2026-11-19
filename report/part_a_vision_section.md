# Section 5 — Vision Agent (CNN Transfer Learning, Architecture Comparison & Grad-CAM)

**Author:** Person A (Vision Specialist)  
**Selected Champion Model:** `resnet50-v1` (`models/vision_resnet50_v1.pt`, **71.51% Test Accuracy / 0.7115 Macro F1**)  
**Compared Architecture:** `efficientnet_v2_s-4fold-tta-v2` (`models/vision_efficientnet_v2_s.pt`, **69.39% Mean CV F1 / 71.28% Peak Fold F1**)

---

## 5.1 Dataset Preparation & Leak-Free Two-Pool Split

To train the Vision Agent while strictly preventing data leakage into the downstream Decision Agent, we used the **Car Damage Severity Dataset** (`1,631` verified collision photos across three canonical classes: `minor`, `moderate`, and `severe`).

Before any training took place, `scripts/prepare_vision_data.py` partitioned the dataset using stratified sampling (`seed=42`) into two completely disjoint pools:
1. **Held-Out Claims Pool (30% — `490` images):** `160` minor, `162` moderate, and `168` severe images (`data/processed/held_out_images.csv`). The CNN **never** trained or validated on these images. They were reserved exclusively for Person C to attach to tabular insurance claims, ensuring that the Decision Agent receives realistic out-of-sample vision probabilities rather than memorized training scores.
2. **CNN Development Pool (70% — `1,141` images):** Stratified into:
   - **Train (`798` images, 70%):** Used for parameter optimization with on-the-fly data augmentation.
   - **Validation (`171` images, 15%):** Used for learning-rate scheduling and checkpoint selection via validation Macro F1.
   - **Test (`172` images, 15%):** Untouched evaluation split used once at the end to report standalone Vision metrics.

---

## 5.2 Model Architectures & Two-Stage Transfer Learning

Following the project guide specification (*"a pretrained ResNet50, EfficientNet or MobileNet with ImageNet weights"*), we implemented and compared two distinct transfer-learning CNN pipelines in `agents/vision_agent.py` and `scripts/train_vision_cnn.py`:

1. **ResNet50 (`resnet50-v1`, $224 \times 224$):**
   - Initialized with `IMAGENET1K_V2` weights; final fully connected head replaced by `Dropout(0.3) -> Linear(2048, 3)`.
   - **Stage 1 (Warmup — 5 epochs):** Convolutional backbone frozen (`requires_grad=False`); head trained with `AdamW` ($\text{lr}=10^{-3}$) and inverse-frequency class weights.
   - **Stage 2 (Fine-Tuning — 12 epochs):** Unfroze `layer3` and `layer4` with `AdamW` ($\text{lr}=10^{-4}$) and `CosineAnnealingLR`.
2. **4-Fold EfficientNet-V2-S Ensemble (`efficientnet_v2_s-4fold-tta-v2`, $288 \times 288$):**
   - Initialized with `IMAGENET1K_V1` weights at higher $288 \times 288$ aspect-preserving resolution across 4 stratified folds (`1,211` deduplicated CV images), trained with Label Smoothing (`0.08`), Mixup ($\alpha=0.2$), RandAugment, layer-wise discriminative learning rates, and evaluated with Horizontal-Flip Test-Time Augmentation (TTA).

---

## 5.3 Quantitative Comparison & Error Analysis

| Model / Configuration | Input Size | Val Macro F1 | Test Accuracy | Test Macro F1 | `minor` F1 | `moderate` F1 | `severe` F1 | Extreme Errors (`minor` → `severe`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50 (Champion)** | $224 \times 224$ | 0.6879 | **71.51%** | **0.7115** | **0.7899** | **0.5688** | **0.7759** | **0 / 56 (0.0%)** |
| **EfficientNet-V2-S (4-Fold + TTA)** | $288 \times 288$ | **0.6939** *(Peak: 0.7128)* | 69.28% | 0.6885 | 0.7458 | 0.5631 | 0.7568 | 1 / 54 (1.9%) |

### Per-Class Performance of the Champion Model (`resnet50-v1`)

| Severity Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Minor** | 0.7460 | 0.8393 | 0.7899 | 56 |
| **Moderate** | 0.5962 | 0.5439 | 0.5688 | 57 |
| **Severe** | 0.7895 | 0.7627 | 0.7759 | 59 |
| **Macro Avg** | **0.7106** | **0.7153** | **0.7115** | **172** |
| **Weighted Avg** | **0.7113** | **0.7151** | **0.7118** | **172** |

### Why Both Architectures Converge to ~70%–71.5% (Confusion Matrix Analysis)
Comparing both confusion matrices (`report/figures/vision_confusion_matrix.png` and `report/figures/vision_confusion_matrix_effnetv2.png`) reveals a critical domain insight:
1. **Near-Zero Extreme Misclassifications (`minor` vs. `severe`):** Both CNNs achieve strong performance on `minor` (`F1 = 0.79`) and `severe` (`F1 = 0.78`). Crucially, **ResNet50 misclassifies 0 out of 56 `minor` photos as `severe`**, and only **2 out of 59 `severe` crashes as `minor`**.
2. **Aleatoric Label Noise in the `moderate` Boundary:** Both ResNet50 and the 4-Fold EfficientNet-V2-S ensemble plateau at `~0.56–0.57 F1` on the intermediate `moderate` class (`47` out of `49` total ResNet50 test errors are single-step neighbor confusions between `minor ↔ moderate` or `moderate ↔ severe`). Because the Kaggle `data3a` images are low-resolution web thumbnails (median size $\approx 275 \times 180$ px) without standardized camera distance or repair cost metadata, the boundary between a large scratch (`minor`) and a small panel dent (`moderate`) is inherently ambiguous.
3. **Why This Is Ideal for Fraud Detection:** In our multi-agent system, the Vision Agent's primary fraud signal is the **severity mismatch feature** ($\text{declared\_rank} - E[\text{photo\_rank}]$), which targets claimants declaring *Major Damage / Total Loss* (`severe`) while submitting a photo of a minor scratch (`minor`). Because `resnet50-v1` has **0% false positives from `minor` to `severe`**, it provides a clean, reliable fraud signal to the Decision Agent.

---

## 5.4 Visual Explainability via Grad-CAM

To ensure a human auditor can verify *why* a photo received a given severity score, we implemented **Grad-CAM** (`explain/gradcam.py`) for both single-CNN (`model.layer4[-1]`) and ensemble (`model.features[-1]`) architectures.

Inspection of `report/figures/vision_gradcam_examples.png` and `report/figures/vision_gradcam_effnetv2.png` confirms that both networks' high-activation regions localize directly onto physical vehicle deformation—such as crumpled hoods, detached front bumpers, dented quarter panels, and burned frames—rather than background pavement, trees, or neighboring cars.

---

## 5.5 Handoff & Limitations

- **Held-Out Pool Scoring:** All `490` held-out photos were scored using `scripts/predict_held_out.py` and exported to `data/processed/vision_predictions.csv` following our standardized `AgentMessage` contract.
- **Limitations:** Single-view exterior photos cannot capture hidden structural or internal mechanical damage, and our synthetic claim-to-photo linking rule is an experimental assumption rather than organic insurer claim data.
