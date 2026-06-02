"""
Q-Learning Reinforcement Learning Module

Implements a Q-Learning agent that learns optimal paths through trial and error.
Agent explores the environment and updates Q-values based on rewards.
"""

import numpy as np
import random
from typing import Tuple, Dict
from environment.grid import OceanGrid


class QLearningAgent:
    """
    Q-Learning agent for path planning.
    
    The agent learns by:
    1. Exploring the environment (taking random actions)
    2. Receiving rewards: +100 for reaching goal, -1 for collision, -1 for each step
    3. Updating Q-values: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
    
    Attributes:
        grid (OceanGrid): The ocean grid
        learning_rate (float): Learning rate (alpha)
        discount_factor (float): Discount factor (gamma)
        epsilon (float): Exploration rate
        q_table (Dict): Q-value table
    """
    
    def __init__(
        self,
        grid: OceanGrid,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            grid: OceanGrid instance
            learning_rate: Learning rate (0-1)
            discount_factor: Discount factor (0-1)
            epsilon: Exploration rate (0-1)
        """
        self.grid = grid
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table: Dict[Tuple[int, int, int, int], Dict[Tuple[int, int], float]] = {}
    
    def state_to_key(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert state to dictionary key.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            State key tuple
        """
        return (x, y)
    
    def get_possible_actions(self, x: int, y: int) -> list:
        """
        Get valid actions from current state.
        
        Actions: 8-directional movement
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            List of valid action deltas
        """
        actions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.grid.is_free(nx, ny):
                    actions.append((dx, dy))
        return actions
    
    def select_action(self, x: int, y: int, training: bool = True) -> Tuple[int, int]:
        """
        Select action using epsilon-greedy strategy.
        
        Args:
            x: X coordinate
            y: Y coordinate
            training: If True, use exploration; if False, use exploitation
            
        Returns:
            Action (dx, dy)
        """
        state = self.state_to_key(x, y)
        possible_actions = self.get_possible_actions(x, y)
        
        if not possible_actions:
            return (0, 0)
        
        # Exploration vs Exploitation
        if training and random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        # Exploitation: choose best action
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in possible_actions}
        
        q_values = self.q_table[state]
        max_q = max(q_values.values()) if q_values else 0
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
    
    def calculate_reward(self, x: int, y: int, goal: Tuple[int, int], collision: bool) -> float:
        """
        Calculate reward for transition.
        
        Rewards:
            +100: Goal reached
            -50: Collision
            -1: Each step
        
        Args:
            x: Current X coordinate
            y: Current Y coordinate
            goal: Goal position
            collision: Whether collision occurred
            
        Returns:
            Reward value
        """
        if collision:
            return -50
        if (x, y) == goal:
            return 100
        return -1  # Small penalty for each step
    
    def learn(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_steps: int = 200
    ) -> Tuple[float, int, list]:
        """
        Train agent for one episode.
        
        Args:
            start: Start position
            goal: Goal position
            max_steps: Maximum steps per episode
            
        Returns:
            (total_reward, steps_taken, path)
        """
        x, y = start
        path = [(x, y)]
        total_reward = 0
        
        for step in range(max_steps):
            if (x, y) == goal:
                total_reward += 100
                break
            
            action = self.select_action(x, y, training=True)
            dx, dy = action
            nx, ny = x + dx, y + dy
            
            collision = not self.grid.is_free(nx, ny)
            if not collision:
                x, y = nx, ny
            
            reward = self.calculate_reward(x, y, goal, collision)
            total_reward += reward
            path.append((x, y))
            
            # Update Q-value
            self._update_q_value(x - dx, y - dy, action, x, y, reward, goal)
        
        return total_reward, len(path), path
    
    def _update_q_value(
        self,
        x: int,
        y: int,
        action: Tuple[int, int],
        next_x: int,
        next_y: int,
        reward: float,
        goal: Tuple[int, int]
    ) -> None:
        """
        Update Q-value using Q-Learning formula.
        
        Args:
            x, y: Current state
            action: Action taken
            next_x, next_y: Next state
            reward: Reward received
            goal: Goal position
        """
        state = self.state_to_key(x, y)
        next_state = self.state_to_key(next_x, next_y)
        
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.get_possible_actions(next_x, next_y)}
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q


if __name__ == "__main__":
    # Test Q-Learning agent
    grid = OceanGrid(50, 50, 5)
    grid.add_obstacle(10, 10, 20, 10)
    
    agent = QLearningAgent(grid, learning_rate=0.1, discount_factor=0.95, epsilon=0.2)
    
    start = (5, 5)
    goal = (45, 45)
    
    print("Training Q-Learning agent...")
    for episode in range(100):
        reward, steps, path = agent.learn(start, goal, max_steps=200)
        if episode % 20 == 0:
            print(f"Episode {episode}: Reward={reward:.1f}, Steps={steps}")
    
    print("\nTraining complete!")
