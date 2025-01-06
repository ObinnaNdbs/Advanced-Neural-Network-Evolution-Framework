# Version 3: Custom CUDA Kernel Integration

## Description
This version integrates a custom CUDA kernel to optimize certain computations in the training pipeline. The CUDA kernel is used for operations like matrix multiplication and activation functions, leveraging GPU acceleration for improved performance.

### Key Features:
- Introduction of a basic CUDA kernel into the PyTorch pipeline.
- Optimized matrix multiplication and activation functions with CUDA.
- Improved GPU memory utilization and execution speed.

## Observations:
1. **Training Loss over Epochs**:
   - The training loss consistently decreased, showcasing effective learning.
   - Started at a higher value and gradually dropped, indicating proper weight updates.

2. **Test Accuracy over Epochs**:
   - Achieved a peak test accuracy of **83.03%**, which is higher compared to Version 1 and Version 2.
   - Fluctuations observed in the later epochs, indicating potential overfitting or insufficient regularization.

3. **CUDA Metrics**:
   - GPU reserved memory remained stable at **417MB**, showcasing efficient memory usage.
   - Total memory allocation and release were well-managed, as indicated by the memory summary logs.

## Comparison with Previous Versions:
- **Accuracy Improvements**:
  - Version 1: 76.00%
  - Version 2: 83.84%
  - Version 3: 83.03% (slightly less than Version 2 but achieved with optimized operations).

- **Training Time**:
  - Reduction in training time due to CUDA kernel acceleration.

## Graphical Analysis:
The attached graph visualizes the training loss and test accuracy over the epochs:
1. **Training Loss**: Shows a consistent decrease, aligning with the model's learning curve.
2. **Test Accuracy**: Peaks at epoch 9 with minor fluctuations.
![Screenshot (652)](https://github.com/user-attachments/assets/4e19900b-1780-4d7e-8930-e70777baaa66)

