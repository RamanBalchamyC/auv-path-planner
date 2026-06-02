"""
Deep Q-Network (DQN) Module

Implements a neural network-based Q-Learning agent using PyTorch.
Provides advanced deep reinforcement learning capabilities (bonus feature).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Tuple, List
from collections import deque
from environment.grid import OceanGrid


class DQNNetwork(nn.Module):
    """
    Deep Q-Network neural network architecture.
    
    Input: State (grid position + goal position)
    Output: Q-values for each action
    """
    
    def __init__(self, input_size: int = 4, hidden_size: int = 128, output_size: int = 8):
        """
        Initialize DQN network.
        
        Args:
            input_size: Input state size (x, y, goal_x, goal_y)
            hidden_size: Hidden layer size
            output_size: Output size (number of actions: 8-directional)
        """
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values for each action
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """
    Deep Q-Network agent for path planning.
    
    Features:
    - Neural network approximation of Q-values
    - Experience replay for stable learning
    - Target network for decorrelated updates
    
    Attributes:
        grid (OceanGrid): The ocean grid
        device (torch.device): CPU or GPU
        network (DQNNetwork): Primary network
        target_network (DQNNetwork): Target network
        optimizer: Adam optimizer
        memory (deque): Experience replay buffer
    """
    
    # Action deltas for 8-directional movement
    ACTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    def __init__(
        self,
        grid: OceanGrid,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        memory_size: int = 10000
    ):
        """
        Initialize DQN agent.
        
        Args:
            grid: OceanGrid instance
            learning_rate: Adam learning rate
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate
            memory_size: Experience replay buffer size
        """
        self.grid = grid
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.memory = deque(maxlen=memory_size)
        
        self.device = torch.device("cpu")  # Use CPU for compatibility
        
        self.network = DQNNetwork(input_size=4, hidden_size=128, output_size=len(self.ACTIONS)).to(self.device)
        self.target_network = DQNNetwork(input_size=4, hidden_size=128, output_size=len(self.ACTIONS)).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def state_to_tensor(self, x: int, y: int, goal_x: int, goal_y: int) -> torch.Tensor:
        """
        Convert state to tensor.
        
        Args:
            x, y: Current position
            goal_x, goal_y: Goal position
            
        Returns:
            State tensor [x, y, goal_x, goal_y]
        """
        state = torch.tensor([x, y, goal_x, goal_y], dtype=torch.float32).to(self.device)
        return state
    
    def select_action(self, x: int, y: int, goal_x: int, goal_y: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy strategy.
        
        Args:
            x, y: Current position
            goal_x, goal_y: Goal position
            training: Use exploration if True
            
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, len(self.ACTIONS) - 1)
        
        state = self.state_to_tensor(x, y, goal_x, goal_y)
        with torch.no_grad():
            q_values = self.network(state)
        return q_values.argmax(dim=0).item()
    
    def remember(self, state: Tuple, action: int, reward: float, next_state: Tuple, done: bool) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state (x, y, goal_x, goal_y)
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode finished flag
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def train_batch(self, batch_size: int = 32) -> float:
        """
        Train network on batch of experiences.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Loss value
        """
        if len(self.memory) < batch_size:
            return 0.0
        
        batch = random.sample(self.memory, batch_size)
        
        states = torch.stack([self.state_to_tensor(*s[0]) for s in batch])
        actions = torch.tensor([s[1] for s in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([self.state_to_tensor(*s[3]) for s in batch])
        dones = torch.tensor([s[4] for s in batch], dtype=torch.float32).to(self.device)
        
        # Current Q-values
        current_q = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q
        
        # Backward pass
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self) -> None:
        """Update target network weights from primary network."""
        self.target_network.load_state_dict(self.network.state_dict())
    
    def learn(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_steps: int = 200
    ) -> Tuple[float, int]:
        """
        Train agent for one episode.
        
        Args:
            start: Start position
            goal: Goal position
            max_steps: Maximum steps per episode
            
        Returns:
            (total_reward, steps_taken)
        """
        x, y = start
        goal_x, goal_y = goal
        total_reward = 0
        
        for step in range(max_steps):
            action_idx = self.select_action(x, y, goal_x, goal_y, training=True)
            dx, dy = self.ACTIONS[action_idx]
            nx, ny = x + dx, y + dy
            
            done = False
            if not self.grid.is_free(nx, ny):
                reward = -50
            elif (nx, ny) == goal:
                reward = 100
                done = True
                nx, ny = goal
            else:
                reward = -1
            
            self.remember((x, y, goal_x, goal_y), action_idx, reward, (nx, ny, goal_x, goal_y), done)
            self.train_batch(batch_size=16)
            
            total_reward += reward
            x, y = nx, ny
            
            if done:
                break
        
        return total_reward, step + 1


if __name__ == "__main__":
    # Test DQN agent
    grid = OceanGrid(50, 50, 5)
    grid.add_obstacle(10, 10, 20, 10)
    
    agent = DQNAgent(grid, learning_rate=0.001, discount_factor=0.99, epsilon=0.2)
    
    start = (5, 5)
    goal = (45, 45)
    
    print("Training DQN agent (this may take a moment)...")
    for episode in range(50):
        reward, steps = agent.learn(start, goal, max_steps=200)
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={reward:.1f}, Steps={steps}")
        if episode % 10 == 9:
            agent.update_target_network()
    
    print("\nTraining complete!")
