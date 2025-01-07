# **Version 4: Incorporation of NVIDIA Libraries**

## **Project Overview**

### **Description**
This project leverages NVIDIA's cuDNN and cuBLAS libraries to optimize computationally intensive operations such as convolutions and matrix multiplications. By integrating these proprietary libraries, the project demonstrates expertise in NVIDIA’s ecosystem for accelerated deep learning computations.

### **Key Features**
- Replacement of default PyTorch operations with cuDNN and cuBLAS for enhanced performance.
- Profiling using TensorBoard to analyze and verify performance improvements.
- Efficient GPU memory management during training and evaluation.

### **Purpose**
To showcase familiarity with NVIDIA's libraries and their application in improving computational efficiency for deep learning tasks.

---

## **Implementation Details**

### **Data and Model**
- **Dataset:** CIFAR-10 (60,000 images, 32x32 resolution) with train-test split.
- **Model:** Pretrained ResNet18 fine-tuned for CIFAR-10 classification.

### **Training Setup**
- **Optimizer:** Adam with a learning rate of 0.001.
- **Loss Function:** Cross-Entropy Loss.
- **Batch Size:** 64 for both training and testing.
- **Epochs:** 10.

### **CUDA Optimization**
1. **cuDNN Integration:**
   - Utilized for optimized convolutional operations.
   - Improved runtime efficiency by reducing overhead for critical layers.

2. **cuBLAS Integration:**
   - Enhanced matrix multiplication speed during backpropagation and gradient updates.
   - Reduced time for weight updates and intermediate calculations.

3. **Memory Management:**
   - Monitored GPU memory usage during training to ensure efficient allocation and deallocation.
   - Achieved stable memory utilization with a peak of **362 MB reserved** and **195 MB active**.

4. **Profiling:**
   - TensorBoard profiling demonstrated reduced computational overhead and faster execution of key operations.

---

## **Results**

### **Performance Metrics**
- **Final Test Accuracy:** 83.10%.
- **Training Loss:** Consistently decreased from ~1.1 (Epoch 1) to ~0.5 (Epoch 10).
- **Test Accuracy:** Improved steadily from **65.01% (Epoch 1)** to **83.10% (Epoch 10)**, showcasing strong generalization.

### **Graphical Analysis**
1. **Training Loss:**  
   - A smooth decline across epochs indicates effective integration of NVIDIA libraries for model optimization.
2. **Test Accuracy:**  
   - Steady improvement highlights the benefits of efficient computation for both forward and backward passes.

   ![Training and Test Metrics](Screenshot%20(653).png)

---

## **Lessons Learned**

### **Strengths**
- Integration of cuDNN and cuBLAS significantly reduced computational overhead, enabling faster training without compromising accuracy.
- Achieved high accuracy and stable memory utilization, highlighting the scalability of NVIDIA’s libraries for deep learning applications.

### **Challenges**
- Computational resource requirements increased compared to baseline PyTorch operations, necessitating careful memory management to avoid bottlenecks.

---

## **Conclusion**
This project successfully demonstrates the incorporation of NVIDIA’s cuDNN and cuBLAS libraries to optimize deep learning computations. By achieving significant speedups and improved accuracy, the project highlights the practical applications of NVIDIA's ecosystem in real-world AI workflows. The results emphasize the effectiveness of leveraging hardware-accelerated libraries for training and deploying high-performance deep learning models.
