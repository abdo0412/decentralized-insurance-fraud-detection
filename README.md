# Project 7 — Decentralized Insurance Fraud Detection

Multi-agent system that audits car insurance claims using collision photos (**Vision Agent**), written claim reports (**Text Agent**), and structured claim fields (**Tabular / Decision Agent**) to decide whether to **validate** a claim or send it to a **human audit**.

## Repository Layout

```text
fraud-detection/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # Raw Kaggle datasets (not committed)
│   └── processed/            # Leak-free splits, held-out image pool, OOF predictions
├── models/                   # Saved checkpoints (vision_resnet50_v1.pt, etc.)
├── notebooks/                # Exploration and Colab/Kaggle GPU training notebooks
├── scripts/
│   ├── prepare_vision_data.py  # Step 1 (Part A): Disjoint split (CNN pool vs Held-out pool)
│   ├── train_vision_cnn.py     # Step 2 (Part A): Two-stage CNN transfer learning + eval
│   └── predict_held_out.py     # Step 3 (Part A): Batch inference on linked claims for Person C
├── agents/
│   ├── base.py               # Shared JSON message envelope & severity mappings
│   ├── vision_agent.py       # Part A: ResNet50 / MobileNetV3 severity classifier
│   ├── text_agent.py         # Part B: Naive Bayes & DistilBERT narrative scorer
│   ├── tabular_agent.py      # Part C: Tabular risk scorer
│   └── decision_agent.py     # Part C: Random Forest meta-classifier + cost threshold
├── orchestrator/
│   └── coordinator.py        # Routes claim to specialist agents & Decision Agent
├── explain/
│   └── gradcam.py            # Part A: Grad-CAM heatmap generator for CNN explainability
├── app/
│   └── streamlit_app.py      # Interactive demo application
├── tests/
│   └── test_vision_agent.py  # Unit tests for message contract & Vision Agent
└── report/
    └── figures/              # Confusion matrices, Grad-CAM grids, SHAP plots
```

## Quickstart — Part A (Vision Agent)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Leak-Free Image Splits (Day 1)
Download the Kaggle **Vehicle Damage Image Dataset** into `data/raw/vehicle_damage/` (organized with severity subfolders such as `01-minor`, `02-moderate`, `03-severe`, or `minor`, `moderate`, `severe`), then run:

```bash
python scripts/prepare_vision_data.py --raw-dir data/raw/vehicle_damage --output-dir data/processed
```
This creates:
- `data/processed/held_out_images.csv` (30% of images — **hand this file to Person C** for linking to claims; never used to train the CNN).
- `data/processed/cnn_splits/train.csv`, `val.csv`, `test.csv` (70% of images — used exclusively for CNN training and evaluation).

### 3. Train & Evaluate the CNN (Day 2–3)
```bash
python scripts/train_vision_cnn.py \
    --splits-dir data/processed/cnn_splits \
    --backbone resnet50 \
    --epochs-warmup 5 \
    --epochs-finetune 10 \
    --batch-size 32
```
Saves the best checkpoint to `models/vision_resnet50_v1.pt` and outputs test metrics + confusion matrix + Grad-CAM examples to `report/figures/`.

### 4. Generate Predictions on Linked Claims for Person C (Day 4)
Once Person C provides `data/processed/linked_claims.csv` (`claim_id,image_path,declared_severity`), run:
```bash
python scripts/predict_held_out.py \
    --claims-csv data/processed/linked_claims.csv \
    --weights models/vision_resnet50_v1.pt \
    --output-csv data/processed/vision_predictions.csv
```

### 5. Run the Demo App
```bash
streamlit run app/streamlit_app.py
```
