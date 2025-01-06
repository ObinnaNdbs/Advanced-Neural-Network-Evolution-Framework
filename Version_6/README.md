# Version 6: Advanced Reinforcement Learning and Optimized Performance

## Project Overview

**Description:**  
This project showcases cutting-edge reinforcement learning (RL) applications aligned with NVIDIA's focus on 3D rendering and advanced agent training in complex environments. The implementation features a reinforcement learning agent trained in the Mujoco-based Humanoid-v4 environment, complemented by 3D visualization techniques to analyze and present learning progression.

**Key Features:**
- Implementation of a Deep Q-Network (DQN) for continuous action spaces using PyTorch.
- Experience Replay with normalized rewards for stable and efficient learning.
- Target Network updates and gradient clipping for enhanced training stability.
- Deployment of ONNX for optimized inference on CUDA-enabled devices, achieving real-time performance.
- Interactive 3D plots to visualize learning progress and dynamics.

**Purpose:**  
Demonstrate advanced reinforcement learning methodologies and interactive 3D visualizations, showcasing their potential for NVIDIA's cutting-edge applications in AI and 3D technologies.

---

## Implementation Details

### Environment Setup
- **Framework:** Mujoco-based Humanoid-v4 environment from OpenAI Gym.
- **Device:** CUDA-enabled GPUs for accelerated computations.

### DQN Architecture

**Network Structure:**
- Fully connected neural network with two hidden layers of 512 neurons each.
- Activation function: ReLU.
- Output layer tailored for continuous action spaces.

**Target Network:**
- Soft updates with a tau value of 0.005.

**Optimization:**
- Adam optimizer with a learning rate of 0.0003.
- Loss function: Mean Squared Error (MSE).
- Gradient clipping with a max norm of 1.0.

### Training Pipeline
- **Episodes:** 1,000 episodes, each with a maximum of 200 steps.
- **Reward Normalization:** Applied to stabilize training dynamics and improve convergence.
- **Exploration Strategy:**
  - Epsilon-greedy exploration.
  - Epsilon decay: 0.995 per episode, with a minimum of 0.01.
- **Experience Replay Buffer:**
  - Buffer size: 100,000 experiences.
  - Batch size: 64.

### ONNX Conversion and Inference Optimization
- Trained DQN model converted to ONNX format for efficient and scalable inference.
- Inference speedup evaluated using ONNX Runtime with CUDAExecutionProvider.
- Observed inference time: 0.1224 seconds for 1,000 runs, showcasing suitability for real-time applications.

---

## Results

### Learning Performance
- **Reward Progression:** The total reward per episode steadily increased as training progressed, with significant improvement beyond 200 episodes.
- **Final Reward:** Total reward peaked at 454.41 during episode 348, demonstrating the agent's capability to master the Humanoid-v4 environment.

### Output Visualization

1. **2D Reward Plot:**
   ![Screenshot (654)](https://github.com/user-attachments/assets/e22455ed-d9c5-43a2-bb7e-1a874115e852)

   - Highlights the convergence of learning performance over episodes.

2. **3D Reward Progression:**
   ![newplot](https://github.com/user-attachments/assets/f15d31a1-7ed1-4c19-a36e-f817c9e7cf8c)

   - Visualizes the correlation between steps and rewards achieved.

3. **3D Learning Progress Visualization:**
  ![newplot (2)](https://github.com/user-attachments/assets/068c553b-01ca-4352-89f4-bda66cd84736)

   - Provides insights into learning patterns and stability over time.


---

## Key Observations

### Learning Progress and Stability

- **Improvement Over Episodes:**
  - The DQN model showed a clear upward trend in total rewards over episodes, as seen in the first graph ("DQN Learning Performance on Humanoid-v4"). Despite noise from training variability, the agent's learning progress is evident.
  - Oscillations in rewards reflect the complexity of the Humanoid-v4 environment's continuous, high-dimensional action space. Nevertheless, the model converged toward higher rewards.

- **Baseline Stability:**
  - Total rewards peaked around 450, while lower bounds stabilized above 100 in later episodes, showing the model avoided poor policies.
  - Early reward fluctuations (episodes 0-300) highlight the exploration phase, transitioning into reduced variability as the model shifted to exploitation, leveraging learned strategies.

### Reward Variability

- The 3D scatter plot reveals reward progression across episodes. Early training had scattered rewards, while later episodes showed clustering in higher ranges, reflecting improved performance.
- The surface plot highlights consistent reward accumulation in later stages, with a nearly flat, high-reward surface in final episodes, demonstrating stability and optimization.

### ONNX Optimization

- ONNX-enabled inference provided significant computational efficiency, achieving impressive speedups. This demonstrates the model's suitability for real-time applications requiring high performance.

### Challenges Overcome

- The Humanoid-v4 environment's high-dimensional action space posed significant challenges. Experience replay, epsilon-greedy exploration, and reward normalization were effectively tailored to address these.
- The model's handling of reward variability underscores the robustness of the policy network and the effectiveness of the applied training strategies.


---

## Conclusion

This project marks a significant advancement in reinforcement learning, demonstrating the successful implementation of a DQN agent in the challenging Humanoid-v4 environment. By addressing the complexities of high-dimensional, continuous action spaces, the project showcased the power of advanced learning techniques such as reward normalization, target network updates, and gradient clipping. These approaches enabled the agent to achieve stable, high-performance policies, overcoming variability and maximizing learning efficiency.
