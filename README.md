# Hypergraph-based Global-Local Virtual Unified Emotion Fusion for Multimodal Emotion Recognition in Conversations
Pytorch implementation for the paper **Hypergraph-based Global-Local Virtual Unified Emotion Fusion for Multimodal Emotion Recognition in Conversations** (Pattern Recognition)

## 📌 Overview
This work proposes a **VUEMO** framework, a global-to-local hypergraph neural network with Virtual Unified Emotion (VUE) modality fusion for multimodal emotion recognition in conversations (MERC). The model addresses two key limitations of existing methods:
- Emotional conflict between cross-modal connections
- Insufficient global interaction ability of GNN-based models

VUEMO integrates a **Global Attention Interaction (GAI)** module for long-range temporal alignment and a **VUE-guided Local Graph Learning (LGL)** module for structured multimodal coordination, achieving SOTA performance on IEMOCAP and MELD datasets.

## 🛠️ Requirements
```bash
Python == 3.11.7
torch == 2.0.1
torch-geometric == 1.7.2
CUDA == 11.3
numpy == 1.26.4
pandas == 2.2.1
scikit-learn == 1.4.1
matplotlib == 3.8.4
ipdb == 0.13.13
```
## 📁 Dataset Preparation
### Pre-extracted Features
We use pre-extracted multimodal features for training (consistent with the paper):
- Textual Features: RoBERTa-Large (1024-dim) / GloVe
- Visual Features: 3D-CNN (IEMOCAP) / DenseNet (MELD) (512-dim)
- Acoustic Features: openSMILE (512-dim)

### Dataset Structure
Place the downloaded features in the following directory structure:
```
VUEMO/
├── IEMOCAP_features.pkl
├── iemocap_features_roberta.pkl
├── MELD_features/
│   ├── MELD_features_raw1.pkl
│   └── meld_features_roberta.pkl
├── model.py
├── train.py
├── dataloader.py
└── ...
```

## 🏃 Training
### Train on IEMOCAP
```
python train.py --Dataset IEMOCAP --base-model GRU --graph_type hyper --modals avl --mm_fusion_mthd concat_DHT --num_L 8 --num_K 4 --lr 1e-4 --batch-size 16 --epochs 100 --use_speaker True
```
### Train on MELD
```
python train.py --Dataset MELD --base-model GRU --graph_type hyper --modals avl --mm_fusion_mthd concat_DHT --num_L 8 --num_K 4 --lr 2e-5 --batch-size 16 --epochs 100 --use_speaker True
```
## 📊 Main Results
### IEMOCAP Dataset (Weighted Acc/F1)
| Metric | VUEMO (Ours) | SOTA Baseline | Improvement |
|--------|--------------|---------------|-------------|
| W-Acc  | 72.47%       | 71.30%        | +1.17%      |
| W-F1   | 72.43%       | 70.10%        | +2.33%      |

### MELD Dataset (Weighted Acc/F1)
| Metric | VUEMO (Ours) | SOTA Baseline | Improvement |
|--------|--------------|---------------|-------------|
| W-Acc  | 69.12%       | 67.78%        | +1.34%      |
| W-F1   | 67.77%       | 66.85%        | +0.92%      |
