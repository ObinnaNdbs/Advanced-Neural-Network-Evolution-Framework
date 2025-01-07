# **Version 5: Inference Optimization with TensorRT**

## **Project Overview**

### **Description**
This project demonstrates the conversion and optimization of a trained model using TensorRT to achieve high-performance inference. The key focus is on enhancing inference speed and efficiency while maintaining model accuracy.

### **Key Features**
- Conversion of ResNet18 from PyTorch to ONNX, followed by TensorRT optimization.
- Performance benchmarking to compare inference times before and after TensorRT optimization.
- Use of FP16 precision in TensorRT for improved computational speed without significant accuracy loss.

### **Purpose**
To showcase expertise in inference optimization techniques critical for real-world deployment in computationally intensive environments.

---

## **Implementation Details**

### **Data and Model**
- **Dataset:** CIFAR-10 with data augmentation techniques, including random horizontal flips and cropping.
- **Model:** Pre-trained ResNet18, modified for CIFAR-10 with a new fully connected layer for 10 output classes.

### **Training and Evaluation**
- **Training Parameters:**
  - Optimizer: Adam
  - Learning Rate: 0.001 with step decay
  - Loss Function: Cross-Entropy Loss
  - Number of Epochs: 10
- **Performance Metrics:**
  - Training Loss: Steadily decreased over epochs (graph attached).
  - Test Accuracy: Reached 83.17% after 10 epochs, indicating effective learning and generalization.

### **Optimization Workflow**
1. **ONNX Conversion:**
   - The trained PyTorch model was exported to ONNX format with static batch size for better compatibility.
   - Validated the ONNX model for structural integrity.

2. **TensorRT Optimization:**
   - Parsed the ONNX model with TensorRT to build an optimized inference engine.
   - Enabled FP16 precision for reduced computation time.
   - Saved the TensorRT engine for efficient deployment.

3. **Inference Benchmarking:**
   - Compared ONNX Runtime and TensorRT for inference speed using the CIFAR-10 test set.
   - Measured end-to-end inference times for both frameworks.

---

## **Results**

### **Performance Comparison**
- **Accuracy:**
  - No loss in accuracy observed during conversion and optimization processes.
- **Inference Speed:**
  - ONNX Inference Time: 17.26 seconds
  - TensorRT Inference Time: 6.75 seconds
  - **Speedup:** TensorRT achieved a **2.56x speedup** over ONNX.

### **Training and Test Metrics**
- **Training Loss:** Demonstrated a smooth decline over 10 epochs, indicating consistent learning.
- **Test Accuracy:** Improved steadily across epochs, reaching 83.17%.
![Screenshot (656)](https://github.com/user-attachments/assets/d2cc6e03-65c4-4662-bd9f-b8883687f654)

### **Visual Representation**
1. **Training Loss over Epochs:** 
   - Demonstrates model convergence during training.
2. **Test Accuracy over Epochs:** 
   - Highlights the progressive improvement in accuracy.



---

## **Conclusion**
This project underscores the effectiveness of TensorRT in optimizing inference for deep learning models. By leveraging FP16 precision and TensorRT's advanced optimizations, significant speedups were achieved without compromising accuracy. These advancements establish TensorRT as a powerful tool for real-world deployment of AI applications requiring high efficiency and performance.
