# Version 1: Basic Neural Network (Baseline)

## Key Results

### Model Performance
- **Dataset**: CIFAR-10 (10 classes, 60,000 images, 32x32 resolution).
- **Model**: A simple CNN with 3 convolutional layers, max pooling, and 2 fully connected layers.
- **Optimizer**: Adam Optimizer with a learning rate of 0.001.
- **Loss Function**: Cross-Entropy Loss.
- **Batch Size**: 128 (training) and 100 (testing).
- **Epochs**: 10.

### Performance Metrics
1. **Final Test Accuracy**: ~76% after 10 epochs.
2. **Training Loss Trend**:
   - Decreased consistently from ~1.5 to ~0.2 over 10 epochs.
   - Indicates effective learning of patterns in the dataset.
3. **Test Accuracy Trend**:
   - Improved steadily from 57.15% (Epoch 1) to a peak of 76.00% (Epoch 8).
   - Slight decline to 74.90% in Epoch 10, suggesting potential overfitting.

### Graph Analysis
- **Training Loss**: Shows a steady and significant decrease, confirming the model's learning capability.
- **Test Accuracy**: Demonstrates consistent improvement, stabilizing near the end of training.

![Screenshot (650)](https://github.com/user-attachments/assets/a0cbda09-4c7e-4c79-9358-f275f90b302a)


---

## Hardware Utilization
- **Device**: NVIDIA GPU (used CUDA for acceleration).
- **GPU Memory**: Peak usage of ~731 MB during training.
- **Efficiency**: No CUDA Out-of-Memory (OOM) errors; efficient GPU utilization.

---

## Lessons Learned
1. **Strengths**:
   - Simple CNN architecture provided a strong baseline.
   - Clear improvement in test accuracy validates the model's design and hyperparameters.

2. **Challenges**:
   - Slight overfitting observed towards the end of training (accuracy declined after epoch 8).
   - Basic normalization without augmentation limits the model's ability to generalize.

---

## Future Directions
### Building on Version 1
- **Version 2: Data Augmentation and Pretrained Model**
  - Incorporate data augmentation techniques like random cropping and horizontal flipping to improve generalization.
  - Leverage pretrained ResNet18 for transfer learning to enhance performance and reduce training time.
- **Expected Benefits**:
  - Mitigate overfitting by increasing data diversity.
  - Boost test accuracy beyond 76% by utilizing more advanced models.

---
