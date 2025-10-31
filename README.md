
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

