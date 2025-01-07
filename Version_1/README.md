# **Version 1: Basic Neural Network (Baseline)**

## **Project Overview**

### **Description**
This project implements a simple Convolutional Neural Network (CNN) using PyTorch for image classification on the CIFAR-10 dataset. As the baseline version, the focus is on establishing a starting point for performance benchmarking without advanced optimization techniques.

### **Key Features**
- Implementation of basic training and testing loops.
- Evaluation of baseline performance metrics, including accuracy and loss.
- Establishes a foundation for future versions with more advanced techniques.

### **Purpose**
To provide a strong starting point for performance comparisons and serve as a benchmark for later improvements.

---

## **Implementation Details**

### **Dataset**
- **CIFAR-10**: Comprises 60,000 32x32 images across 10 classes.

### **Model Architecture**
- **Structure**:
  - 3 convolutional layers with ReLU activation and max pooling.
  - 2 fully connected layers for classification.
- **Optimizer**: Adam Optimizer with a learning rate of 0.001.
- **Loss Function**: Cross-Entropy Loss.
- **Batch Sizes**: 
  - 128 for training.
  - 100 for testing.
- **Training Epochs**: 10.

---

## **Key Results**

### **Performance Metrics**
1. **Final Test Accuracy**: Approximately **76%** after 10 epochs.
2. **Training Loss**:
   - Consistent decrease from ~1.5 to ~0.2 over 10 epochs.
   - Confirms the model's ability to learn patterns effectively.
3. **Test Accuracy**:
   - Improved steadily from **57.15% (Epoch 1)** to a peak of **76.00% (Epoch 8)**.
   - Slight decline to **74.90% (Epoch 10)** suggests potential overfitting.

### **Graph Analysis**
- **Training Loss**: Steady and significant decline over epochs, indicating the model's effective learning process.
- **Test Accuracy**: Consistent improvement, stabilizing near the end of training.

![Screenshot (650)](https://github.com/user-attachments/assets/fa01a79a-24f2-4e48-8d77-6d072c28ad65)

---

## **Hardware Utilization**
- **Device**: NVIDIA GPU with CUDA acceleration.
- **GPU Memory**: Peak usage of approximately **731 MB** during training.
- **Efficiency**: 
  - No CUDA Out-of-Memory (OOM) errors.
  - Efficient utilization of GPU resources.

---

## **Lessons Learned**

### **Strengths**
- The simple CNN architecture provided a strong baseline.
- Clear improvement in test accuracy validates the model's design and hyperparameters.

### **Challenges**
- **Overfitting**:
  - Test accuracy declined after Epoch 8, indicating slight overfitting.
- **Generalization**:
  - Lack of data augmentation limited the model's ability to generalize to unseen data.

---

## **Future Directions**

### **Building on Version 1**
1. **Version 2: Data Augmentation and Pretrained Model**
   - Add data augmentation techniques (e.g., random cropping and horizontal flipping) to enhance generalization.
   - Incorporate a pretrained ResNet18 model for transfer learning to improve performance and reduce training time.
2. **Expected Benefits**:
   - Mitigate overfitting by increasing data diversity.
   - Boost test accuracy beyond **76%** by leveraging more advanced architectures.

---
