# Essence of CUDA in Biomedical Image Segmentation Using U-Net Architecture

![Status](https://img.shields.io/badge/status-under%20development-yellow)
![CUDA](https://img.shields.io/badge/CUDA-accelerated-blue)

This project focuses on the role of **CUDA** in accelerating biomedical image segmentation tasks, specifically using the **U-Net architecture** on the **DRIVE dataset**. The project integrates CUDA to leverage GPU parallelism, enabling faster computation and efficient training.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Importance of CUDA](#importance-of-cuda)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview
Retinal image segmentation plays a critical role in diagnosing eye-related diseases. The U-Net architecture has been a standard for biomedical image segmentation due to its encoder-decoder structure and skip connections.

In this project:
- **U-Net** is implemented to segment retinal vessels from images.
- The **DRIVE Dataset** (Digital Retinal Images for Vessel Extraction) serves as the primary dataset.
- CUDA acceleration is used to enhance training and inference speed, making the model practical for real-time applications.

---

## Importance of CUDA
### Why CUDA?
CUDA (Compute Unified Device Architecture) by NVIDIA allows developers to harness the parallelism of GPUs, making it ideal for computationally intensive tasks like deep learning.

### Benefits in this Project
1. **Efficient Data Processing**: CUDA handles large image tensors faster than CPU-based computation.
2. **Parallel Execution**: CUDA performs multiple matrix operations simultaneously, critical for backpropagation and convolution operations in U-Net.
3. **Reduced Training Time**: Training on GPU with CUDA support drastically reduces the time compared to CPU-only setups.
4. **Scalability**: Supports larger batch sizes and higher-resolution images.

### Key Metrics with CUDA
- **Speed-Up Ratio**: GPU-accelerated training is typically 5–20 times faster than CPU training.
- **Memory Utilization**: Optimized with techniques like mixed-precision training.

---

## Dataset
**DRIVE Dataset** is designed for retinal vessel extraction and contains:
- **Training Set**: 20 images with manually annotated vessel masks.
- **Test Set**: 20 images with corresponding masks.

Dataset structure:
- `rgb_images`: Retinal images (RGB format).
- `manual_masks/mask`: Manually annotated vessel masks.

Access dataset via Deep Lake:
```python
import deeplake
train_ds = deeplake.load("hub://activeloop/drive-train")
test_ds = deeplake.load("hub://activeloop/drive-test")
```

---

## Installation
### Clone the Repository
```bash
git clone https://github.com/your-username/cuda-bioimage-segmentation.git
cd cuda-bioimage-segmentation
```

### Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### CUDA Toolkit
Ensure that CUDA Toolkit is installed. Verify GPU availability:
```python
import torch
print(torch.cuda.is_available())
```

---

## Usage
### 1. Training
Run the training script with CUDA support:
```bash
python train.py --epochs 50 --batch_size 8 --lr 0.001 --device cuda
```

### 2. Visualization
Visualize input images, predicted masks, and ground truth masks:
```bash
python visualize.py
```

### 3. Experiment Tracking with MLFlow
Track training metrics and model versions:
```bash
mlflow run . -P epochs=50 -P batch_size=8
```

---

## Pipeline
```plaintext
1. Data Loading:
   - DRIVE Dataset > PyTorch DataLoader
2. CUDA-Accelerated Training:
   - U-Net architecture implemented with PyTorch
   - Mixed Precision Training for speed and memory optimization
3. Experiment Tracking:
   - Metrics logged via MLFlow
4. Visualization:
   - Comparison of input images, predicted masks, and ground truth
```

---

## Results
### Performance Metrics
- **Dice Coefficient**: Measures overlap between predicted and ground truth masks.
- **Jaccard Index**: Measures similarity between sets.
- **Speed-Up**: Training is approximately **10x faster** with CUDA.

| Metric          | Value     |
|------------------|-----------|
| Dice Coefficient | 0.88      |
| Jaccard Index    | 0.79      |
| Speed-Up Ratio   | ~10x      |

### Visualization Example
| Input Image       | Predicted Mask   | Ground Truth Mask |
|--------------------|------------------|-------------------|
| ![Input](images/input.png) | ![Prediction](images/predicted.png) | ![Ground Truth](images/ground_truth.png) |

---

## Future Work
1. **Model Optimization**:
   - Prune the U-Net architecture for edge-device deployment.
   - Explore transfer learning with pre-trained models.

2. **Scalability**:
   - Experiment with larger datasets and higher-resolution images.
   - Deploy trained model on Kubernetes for real-time segmentation.

3. **Explainability**:
   - Analyze feature importance using saliency maps.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature/your-feature`).
3. Commit your changes.
4. Push to the branch and create a pull request.

---

**Contact**: For questions or collaboration, email `aryan2002pandeythegrt@gmail.com`.
