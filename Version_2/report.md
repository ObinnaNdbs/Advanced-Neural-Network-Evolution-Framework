# Version 2: Data Augmentation and Pretrained Model

## Key Results

### Model Performance
- **Dataset**: CIFAR-10 (10 classes, 60,000 images, 32x32 resolution).
- **Model**: ResNet18 pretrained on ImageNet, fine-tuned for CIFAR-10 classification.
- **Optimizer**: Adam Optimizer with a learning rate of 0.001 and learning rate decay scheduler (step size: 10, gamma: 0.1).
- **Loss Function**: Cross-Entropy Loss.
- **Batch Size**: 128 (training) and 100 (testing).
- **Epochs**: 10.

### Data Augmentation
- Applied **random horizontal flips** and **random crops** during training.
- Enhanced data diversity, which improved the model's ability to generalize.

### Performance Metrics
1. **Final Test Accuracy**: ~83.84% after 10 epochs.
2. **Training Loss Trend**:
   - Decreased consistently, stabilizing near the end.
   - Indicates effective learning with reduced overfitting compared to Version 1.
3. **Test Accuracy Trend**:
   - Improved steadily, reaching a peak of 83.84% at Epoch 9.
   - Slight decline to 82.48% at Epoch 10, suggesting minimal overfitting.

### Graph Analysis
- **Training Loss**: Shows a steady decrease, confirming effective learning with augmentation.
- **Test Accuracy**: Demonstrates consistent improvement, surpassing the baseline (Version 1) by ~7.84%.
![Screenshot (651)](https://github.com/user-attachments/assets/abab6f32-0a42-4771-b73e-cd131b9ce170)


---

## Hardware Utilization
- **Device**: NVIDIA GPU (used CUDA for acceleration).
- **GPU Memory**: Peak usage of ~731 MB during training.
- **Efficiency**: No CUDA Out-of-Memory (OOM) errors; efficient GPU utilization.

---

## Lessons Learned
1. **Strengths**:
   - Fine-tuning ResNet18 provided significant improvements over the simple CNN in Version 1.
   - Data augmentation enhanced model generalization and test accuracy.
2. **Challenges**:
   - Slight overfitting observed near the end of training.
   - More diverse augmentation strategies could further boost performance.

---

## Future Directions
- **Version 3: Custom CUDA Kernel Integration**
  - Incorporate custom CUDA kernels to optimize operations like matrix multiplication or activation functions.
  - Focus on demonstrating GPU programming skills and performance gains.

---
