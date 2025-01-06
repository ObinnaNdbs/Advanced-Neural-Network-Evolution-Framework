# Version 5

# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import onnx
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Ensure logs directory exists
os.makedirs('./logs', exist_ok=True)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Data Augmentation and Normalization
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Load Pretrained ResNet18 Model and Modify for CIFAR-10
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training Loop
train_losses = []
test_accuracies = []

def train(epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(trainloader))

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Test Accuracy: {accuracy:.2f}%")

# Train and Evaluate Model
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    train(epoch)
    scheduler.step()
    test()
    print(torch.cuda.memory_summary(device=device, abbreviated=True))

# Save Model
torch.save(model.state_dict(), "resnet18_cifar10.pth")
print("Model saved as resnet18_cifar10.pth")

# Convert Model to ONNX
onnx_filename = "resnet18_cifar10.onnx"
dummy_input = torch.randn(1, 3, 32, 32, device=device)

torch.onnx.export(
    model, dummy_input, onnx_filename,
    export_params=True, opset_version=16,
    do_constant_folding=True,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Model converted to ONNX: {onnx_filename}")

# Load ONNX Model and Run Inference
onnx_model = onnx.load(onnx_filename)
onnx.checker.check_model(onnx_model)
ort_session = ort.InferenceSession(onnx_filename, providers=['CUDAExecutionProvider'])

def onnx_inference(image):
    image = image.unsqueeze(0).numpy()
    ort_inputs = {'input': image}
    return ort_session.run(None, ort_inputs)[0]

# Measure ONNX Inference Time
start_time = time.time()
for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    for img in images:
        onnx_inference(img.cpu())
end_time = time.time()
onnx_time = end_time - start_time
print(f"ONNX Inference Time: {onnx_time:.2f} seconds")

# Convert ONNX to TensorRT
# trt_logger = trt.Logger(trt.Logger.WARNING)
trt_logger = trt.Logger(trt.Logger.VERBOSE)  # ✅ Change to VERBOSE mode
builder = trt.Builder(trt_logger)
network = builder.create_network(1)
parser = trt.OnnxParser(network, trt_logger)

with open(onnx_filename, 'rb') as model_file:
    success = parser.parse(model_file.read())
    for i in range(parser.num_errors):  # ✅ Print all errors
        print(parser.get_error(i))
    if not success:
        raise RuntimeError("Failed to parse ONNX model!")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Failed to build TensorRT engine!")

# Save TensorRT Engine
engine_path = "resnet18_cifar10.trt"
with open(engine_path, "wb") as f:
    f.write(serialized_engine)
print(f"TensorRT Engine saved: {engine_path}")

# Load TensorRT Engine
runtime = trt.Runtime(trt_logger)
with open(engine_path, "rb") as f:
    serialized_engine = f.read()

engine = runtime.deserialize_cuda_engine(serialized_engine)
if engine is None:
    raise RuntimeError("Failed to deserialize TensorRT engine!")

context = engine.create_execution_context()

# Allocate Memory
d_input = cuda.mem_alloc(1 * 3 * 32 * 32 * 4)  # FP32 = 4 bytes
d_output = cuda.mem_alloc(10 * 4)
stream = cuda.Stream()

def trt_inference(image):
    image = image.numpy().astype('float32')
    cuda.memcpy_htod_async(d_input, image, stream)
    context.execute_async_v2([d_input.ptr, d_output.ptr], stream.handle)
    stream.synchronize()
    output = np.empty(10, dtype='float32')  # ✅ Faster than pagelocked_empty
    cuda.memcpy_dtoh_async(output, d_output, stream)
    return output

# Measure TensorRT Inference Time
start_time_trt = time.time()
for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    for img in images:
        trt_inference(img.cpu())
end_time_trt = time.time()
trt_time = end_time_trt - start_time_trt
print(f"TensorRT Inference Time: {trt_time:.2f} seconds")

# Compare Speedup
speedup = onnx_time / trt_time
print(f"TensorRT Speedup over ONNX: {speedup:.2f}x")

# Clean up memory
del context
torch.cuda.empty_cache()

# Plot Training Loss and Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.show()
