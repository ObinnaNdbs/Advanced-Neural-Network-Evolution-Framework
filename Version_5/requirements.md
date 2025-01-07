## **Requirements**

To replicate the setup for Version 5, ensure the following:

### **Hardware**
- NVIDIA GPU with TensorRT support.
- Compute capability 6.0 or higher.

### **Software**
- Python 3.8 or higher.
- PyTorch 1.9 or higher.
- ONNX and ONNX Runtime (for model conversion and validation).
- TensorRT 8.0 or higher.
- PyCUDA for TensorRT engine execution.

### **Dataset**
CIFAR-10 dataset (downloaded automatically through Torchvision).

### **Dependencies**
Install the required Python packages:
```bash
pip install torch torchvision onnx onnxruntime tensorrt pycuda matplotlib
