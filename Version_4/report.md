# Version 4: Incorporation of NVIDIA Libraries

## Description
Leverages NVIDIA’s cuDNN and cuBLAS libraries for optimized operations, including convolutions and matrix multiplications, demonstrating expertise in NVIDIA’s ecosystem for accelerated computation.

## Key Results

### Model Performance
- **Dataset**: CIFAR-10 (60,000 images, 32x32 resolution).
- **Model**: Pretrained ResNet18 fine-tuned for CIFAR-10.
- **Optimizer**: Adam Optimizer with a learning rate of 0.001.
- **Loss Function**: Cross-Entropy Loss.
- **Batch Size**: 64 for training and testing.
- **Epochs**: 10.

### Performance Metrics
1. **Final Test Accuracy**: 83.10%.
2. **Training Loss**:
   - Consistently decreased across epochs, starting at ~1.1 and ending at ~0.5.
3. **Test Accuracy**:
   - Improved from 65.01% (Epoch 1) to 83.10% (Epoch 10), showing steady learning and generalization.

### CUDA Metrics
- **Memory Usage**:
  - Peak GPU memory: 362 MB (reserved), 195 MB (active).
  - Efficient allocation and deallocation observed across epochs.
- **Profiling**:
  - TensorBoard profiling indicates reduced computational overhead for critical operations.

## Graphical Analysis
![Screenshot (653)](https://github.com/user-attachments/assets/b8a2ae46-3ba7-4303-8a9f-252fd5ebe369)

1. **Training Loss**: Smooth decrease over epochs indicates effective learning.
2. **Test Accuracy**: Steady improvement highlights the benefits of optimized operations.

## Lessons Learned
1. **Strengths**:
   - Integration of cuDNN and cuBLAS significantly enhanced computation efficiency.
   - Achieved high accuracy with stable memory utilization.
2. **Challenges**:
   - Increased computational requirements compared to baseline models.

