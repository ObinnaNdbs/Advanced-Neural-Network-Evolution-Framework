# Version 3: Custom CUDA Kernel Integration

## Key Results

### Model Performance
- **Dataset**: CIFAR-10 (10 classes, 60,000 images, 32x32 resolution).
- **Model**: Pretrained ResNet18, modified for CIFAR-10 with a custom final fully connected layer.
- **Optimizer**: Adam Optimizer with a learning rate of 0.001.
- **Loss Function**: Cross-Entropy Loss.
- **Batch Size**: 128 (training) and 100 (testing).
- **Epochs**: 10.

### Performance Metrics
1. **Final Test Accuracy**: ~83.03% after 10 epochs.
2. **Training Loss Trend**:
   - Reduction in training time due to CUDA kernel acceleration.
3. **Test Accuracy Trend**:
   - Improved steadily from 71.08% (Epoch 1) to a peak of 83.03% (Epoch 9).
   - Slight decline to 82.72% in Epoch 10, indicating stabilization rather than overfitting.

### Graph Analysis
- **Training Loss**: Shows a steady and significant decrease, indicating robust learning.
- **Test Accuracy**: Demonstrates consistent improvement, with a stabilization phase towards the end.

![Screenshot (652)](https://github.com/user-attachments/assets/e9ce7776-505e-463b-b010-9c1b0ef0aee7)

---

## Hardware Utilization
- **Device**: NVIDIA GPU (leveraging CUDA for acceleration).
- **GPU Memory**: Peak usage of ~334 MB during training.
- **Efficiency**: No CUDA Out-of-Memory (OOM) errors; effective memory utilization.

---

## Lessons Learned
1. **Strengths**:
   - Pretrained ResNet18 model significantly boosted test accuracy compared to baseline.
   - Custom CUDA kernel integration demonstrated efficient GPU utilization.

2. **Challenges**:
   - Training process reached a plateau in test accuracy after Epoch 9.
   - Resource constraints (limited memory usage) necessitated careful optimization.

---

## Future Directions
### Building on Version 3
- **Version 4: Incorporation of NVIDIA Libraries**
  - Utilize cuDNN and cuBLAS for optimized convolutions and matrix multiplications.
  - Aim to improve training efficiency and reduce computational overhead.

---
