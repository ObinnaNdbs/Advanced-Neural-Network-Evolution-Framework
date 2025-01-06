# Version 6

!pip install gym[mujoco] pygame swig onnx onnxruntime mujoco

import torchvision.transforms as transforms
import onnx
import onnxruntime as ort
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import random
from collections import deque



#env = gym.make("Humanoid-v4", render_mode="human")  # Enable visualization

# ✅ Create Mujoco Environment
env = gym.make("Humanoid-v4")

# ✅ Neural Network for Q-Learning
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ✅ Initialize Model and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # Continuous actions
# model = DQN(state_dim, action_dim).to(device)
policy_network = DQN(state_dim, action_dim).to(device)
target_network = DQN(state_dim, action_dim).to(device)
target_network.load_state_dict(policy_network.state_dict())  # Initialize target net
# ✅ Soft update for target network
tau = 0.005  # Small update step
for target_param, policy_param in zip(target_network.parameters(), policy_network.parameters()):
    target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

optimizer = optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.MSELoss()

target_update_frequency = 100  # Update target network every 100 episodes

# ✅ Experience Replay Buffer
buffer = []
buffer_size = 100000
# ✅ Experience Replay Buffer (Use deque for efficiency)
buffer = deque(maxlen=buffer_size)
batch_size = 64

# ✅ Training Parameters
num_episodes = 1000  # Increase episodes for training
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
reward_history = []

for episode in range(num_episodes):
    
    state = env.reset()  # Remove the unpacking `_`
    # state, _ = env.reset()  # ✅ Mujoco requires unpacking
    total_reward = 0

    for step in range(200):  # Max 200 steps per episode
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # ✅ Epsilon-Greedy Exploration
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore: random action
        else:
            with torch.no_grad():
                # action = model(state_tensor).cpu().numpy().flatten()  # Exploit: use DQN
                action = policy_network(state_tensor).cpu().numpy().flatten()
                action = np.clip(action, env.action_space.low, env.action_space.high)  # ✅ Clip action values
    
        
        # ✅ Take step in the environment
        # next_state, reward, done, _, _ = env.step(action)  
        next_state, reward, done, info = env.step(action)

        # ✅ Normalize rewards
        reward_mean = np.mean([r for (_, _, r, _, _) in buffer])
        reward_std = np.std([r for (_, _, r, _, _) in buffer]) + 1e-5
        normalized_reward = (reward - reward_mean) / reward_std


        # ✅ Store in Experience Replay Buffer
        #buffer.append((state, action, reward, next_state, done))
        #if len(buffer) > buffer_size:
        #    buffer.pop(0)  # Remove oldest experience
        # from collections import deque
        # buffer = deque(maxlen=buffer_size)
        buffer.append((state, action, normalized_reward, next_state, done))


        # ✅ Move to next state
        state = next_state  
        total_reward += reward
        
        if done:
            break

    # ✅ Train the Model using Experience Replay
    if len(buffer) > batch_size:
        minibatch = random.sample(buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=device)  # ✅ Keep as float for Mujoco
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device)

        # ✅ Directly use model predictions for continuous action environments like Mujoco
        # ✅ Get all Q-values
        # q_values = model(states)  
        q_values = policy_network(states)
        q_values = torch.sum(q_values * actions, dim=1, keepdim=True)


        # ✅ Extract Q-values for the actions that were actually taken
        q_values = torch.sum(q_values * actions, dim=1, keepdim=True)  

        # ✅ Get the max Q-value for the next state
        # next_q_values = model(next_states).max(dim=1, keepdim=True)[0]  

        # ✅ Get next state Q-values from target network
        with torch.no_grad():
            next_q_values = target_network(next_states).max(dim=1, keepdim=True)[0]

        # ✅ Ensure expected_q_values matches q_values shape
        expected_q_values = rewards.unsqueeze(1) + gamma * next_q_values * (1 - dones).unsqueeze(1)  

        loss = criterion(q_values, expected_q_values.detach())  # ✅ Keep loss computation the same
        optimizer.zero_grad()
        loss.backward()

        # ✅ Clip gradients
        torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)

        optimizer.step()

    # ✅ Update target network periodically
    if episode % target_update_frequency == 0:
        target_network.load_state_dict(policy_network.state_dict())


    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay Exploration Rate
    reward_history.append(total_reward)
    
    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

    


# ✅ Plot Learning Performance
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Learning Performance on Humanoid-v4")
plt.show()

# ✅ Save Trained Model
torch.save(model.state_dict(), "dqn_humanoid.pth")
print("Model saved as dqn_humanoid.pth")

# ✅ Convert Model to ONNX for Optimized Inference
onnx_filename = "dqn_humanoid.onnx"
dummy_input = torch.randn(1, state_dim).to(device)

torch.onnx.export(model, dummy_input, onnx_filename, 
                  export_params=True, opset_version=11, 
                  input_names=['state'], output_names=['action'])

print(f"Model converted to ONNX: {onnx_filename}")

# ✅ ONNX Inference Performance Testing
onnx_model = onnx.load(onnx_filename)
onnx.checker.check_model(onnx_model)
ort_session = ort.InferenceSession(onnx_filename, providers=['CUDAExecutionProvider'])

def onnx_inference(state):
    ort_inputs = {'state': state.numpy()}
    return ort_session.run(None, ort_inputs)[0]

# ✅ Measure ONNX Speedup
start_time = time.time()
for _ in range(1000):
    state = torch.randn(1, state_dim)
    onnx_inference(state)
end_time = time.time()
print(f"ONNX Inference Time: {end_time - start_time:.4f} seconds")


import plotly.graph_objects as go

# Convert reward history into a 3D surface (Episode vs Step vs Reward)
x = list(range(len(reward_history)))
y = list(range(200))  # Assuming 200 steps per episode
z = np.array([reward_history for _ in range(200)])  # Repeat for visualization

# Create 3D surface plot
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(
    title="3D Visualization of Humanoid-v4 Learning Progress",
    scene=dict(
        xaxis_title="Episode",
        yaxis_title="Steps",
        zaxis_title="Reward",
    )
)

# Show in Colab
fig.show()

# Convert reward history into a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=list(range(len(reward_history))),
    y=reward_history,
    z=[i for i in range(len(reward_history))], 
    mode='markers'
)])
fig.update_layout(
    title="3D Visualization of Reward Progression"
 
    )
fig.show()
