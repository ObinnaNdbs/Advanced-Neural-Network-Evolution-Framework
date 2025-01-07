# **Version 2: Data Augmentation and Pretrained Model**

## **Project Overview**

### **Description**
This version focuses on enhancing model performance by leveraging data augmentation techniques and transfer learning with a pretrained ResNet18 model. The integration of horizontal flips and random crops during training improved data diversity, enabling better generalization to unseen data.

### **Key Features**
- **Data Augmentation:**
  - Introduced random horizontal flips and random cropping during training to create a diverse dataset.
- **Transfer Learning:**
  - Fine-tuned the ResNet18 model pretrained on ImageNet for the CIFAR-10 classification task.

### **Purpose**
To demonstrate how data augmentation and transfer learning can significantly improve model accuracy and robustness, building upon the baseline established in Version 1.

---

## **Implementation Details**

### **Data and Model**
- **Dataset:** CIFAR-10, containing 10 classes, 60,000 images, each 32x32 pixels.
- **Model:** Pretrained ResNet18 modified with a fully connected layer for 10-class classification.

### **Training Pipeline**
- **Data Augmentation Techniques:**
  - Applied random horizontal flips and random cropping to diversify training samples.
- **Training Configuration:**
  - Optimizer: Adam with an initial learning rate of 0.001 and a step decay scheduler.
  - Loss Function: Cross-Entropy Loss.
  - Batch Size: 128 for training, 100 for testing.
  - Number of Epochs: 10.

---

## **Results**

### **Model Performance**
1. **Final Test Accuracy:**
   - Achieved **83.84%** after 10 epochs, surpassing the baseline from Version 1 by approximately **7.84%**.
2. **Training Loss Trend:**
   - Demonstrated a consistent decline, stabilizing toward the final epochs.
3. **Test Accuracy Trend:**
   - Steadily improved, peaking at Epoch 9. Minor decline observed in Epoch 10 due to slight overfitting.

### **Graph Analysis**
- **Training Loss:**
  - Indicates effective learning and reduced overfitting compared to Version 1.
- **Test Accuracy:**
  - Shows substantial improvement, confirming the effectiveness of data augmentation and transfer learning.
  
![Screenshot (651)](https://github.com/user-attachments/assets/44a6b43b-0870-452d-820c-6063af8862be)

---

## **Hardware Utilization**
- **Device:** NVIDIA GPU with CUDA acceleration.
- **GPU Memory:** Peak usage of ~731 MB during training.
- **Efficiency:** Smooth training without CUDA OOM errors, demonstrating efficient utilization of resources.

---

## **Lessons Learned**
1. **Strengths:**
   - Transfer learning with ResNet18 significantly outperformed the custom CNN baseline in Version 1.
   - Data augmentation effectively increased model generalization and reduced overfitting.
2. **Challenges:**
   - Slight overfitting observed near the end of training, suggesting the need for additional regularization.
   - Incorporating more diverse augmentation techniques could further enhance model performance.

---

## **Conclusion**
This version demonstrated the power of transfer learning and data augmentation in improving model performance. Fine-tuning ResNet18 and applying augmentation techniques resulted in a robust model with a test accuracy of **83.84%**, a substantial improvement over the baseline established in Version 1.

**Next Steps:**
- Explore custom CUDA kernel integration in **Version 3** to optimize key operations and demonstrate advanced GPU programming techniques.
