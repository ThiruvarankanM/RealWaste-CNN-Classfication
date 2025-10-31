
# RealWaste CNN Image Classification

**EN3150 Assignment 03 - Pattern Recognition**  
A comprehensive deep learning implementation for multi-class waste image classification using Convolutional Neural Networks.

---

##  Dataset Overview

**RealWaste Dataset** from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste)
- **Total Images**: 4,752 images (224×224 pixels)
- **Classes**: 9 waste categories
  - Cardboard, Glass, Metal, Organic, Paper, Plastic, Textile, Trash, Wood
- **Data Split**: 70% Training (3,326) | 15% Validation (712) | 15% Testing (712)
- **Download**: [https://archive.ics.uci.edu/dataset/908/realwaste](https://archive.ics.uci.edu/dataset/908/realwaste)
---

##  Architecture & Models

### 1. Custom CNN (From Scratch)
**Progressive architecture with hierarchical feature learning**

| Component | Configuration |
|-----------|---|
| Conv Blocks | 4 blocks with filters: 32 → 64 → 128 → 256 |
| Kernel Size | 3×3 (Industry standard) |
| Pooling | 2×2 MaxPooling after each block |
| Regularization | Batch Norm + Progressive Dropout (0.25→0.50) |
| Dense Layer | 256 units with ReLU activation |
| Output | Softmax (9 classes) |
| **Total Parameters** | **13.43M** |

**Performance:**
- Test Accuracy: **71.45%** ✓
- Validation Accuracy: **73.30%**
- Test Loss: 0.8587
- Training Accuracy: 79.83%

### 2. Transfer Learning Models

| Model | Accuracy | Parameters | Type |
|-------|----------|------------|------|
| **MobileNetV2** | **76.56%** | 2.59M | Best Efficiency |
| Custom CNN | 71.45% | 13.43M | From Scratch |
| VGG16 | 68.47% | 14.85M | Baseline |

---

##  Key Experimental Findings

### Learning Rate Analysis
Systematic testing with three learning rates over 15 epochs:

| Learning Rate | Best Val Acc | Status |
|---|---|---|
| 0.0001 | 34.52% | Too Conservative |
| **0.001** | **64.06%** | ✓ **Optimal** |
| 0.01 | 44.18% | Too Aggressive |

**Selected**: 0.001 (balanced convergence speed and stability)

### Optimizer Comparison
Impact of momentum in SGD-based optimization:

| Optimizer | Test Accuracy | Observation |
|---|---|---|
| SGD (No Momentum) | 11.79% | ❌ Failed (random guessing) |
| **SGD + Momentum** | **51.42%** | +336% improvement |
| Adam | 73.30% | ✓ **Best long-term** |

**Key Insight**: Momentum is essential for non-convex optimization; Adam provides superior adaptive per-parameter scaling.

### Training Configuration
```
Optimizer:  Adam (lr=0.001, β₁=0.9, β₂=0.999)
Loss:       Sparse Categorical Cross-Entropy
Batch Size: 32
Epochs:     50 (Early stopping at epoch 42)
Callbacks:  ReduceLROnPlateau + EarlyStopping
```

---

##  Activation Functions

| Layer | Function | Reason |
|-------|----------|--------|
| Hidden Layers | ReLU | Avoids vanishing gradients, computationally efficient |
| Output Layer | Softmax | Multi-class probability distribution |

---

##  Key Results Summary

### Custom CNN Training
- **Best Validation Accuracy**: 73.30%
- **Test Accuracy**: 71.45%
- **Generalization Gap**: 6.53% (controlled overfitting)
- **Training-Validation Balance**: Effective regularization achieved

### Performance Insights
✓ Progressive filter expansion captures hierarchical features efficiently  
✓ Batch normalization stabilizes training and enables higher learning rates  
✓ Progressive dropout (0.25→0.50) prevents overfitting in deeper layers  
✓ Adam optimizer with LR=0.001 provides stable convergence  

---

## 👥 Team ByteBrains

- S. Abisan (220013N)
- S. Changeethan (220084F)
- S. Pingalan (220478R)
- M. Thiruvarankan (220647K)

---

##  Citation

Single, S., Iranmanesh, S., & Raad, R. (2023). *RealWaste* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G

---

##  License

This project is part of EN3150 Pattern Recognition course assignment.

**Repository**: [RealWaste-CNN-Classfication](https://github.com/ThiruvarankanM/RealWaste-CNN-Classfication)
