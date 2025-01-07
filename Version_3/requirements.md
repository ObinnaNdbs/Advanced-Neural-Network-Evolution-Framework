## **Requirements**

To replicate the setup for Version 3, ensure the following:

### **Hardware**
- NVIDIA GPU with CUDA support.
- Compute capability 6.0 or higher (required for custom CUDA kernels).

### **Software**
- Python 3.8 or higher.
- PyTorch 1.9 or higher.
- Torchvision 0.10 or higher.
- CUDA Toolkit 11.3 or higher.
- NVIDIA Nsight Compute (optional for debugging and profiling custom kernels).

### **Dataset**
CIFAR-10 dataset (downloaded automatically through Torchvision).

### **Dependencies**
Install the required Python packages:
```bash
pip install torch torchvision pycuda matplotlib
