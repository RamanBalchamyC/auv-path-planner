"""
Obstacle Management Module

Handles placement and management of obstacles in the ocean grid.
Simulates rocks, structures, and other underwater hazards.
"""

from typing import List, Tuple
from dataclasses import dataclass
import random


@dataclass
class Obstacle:
    """Represents a single rectangular obstacle."""
    x: int
    y: int
    width: int
    height: int
    name: str = "Rock"


class ObstacleManager:
    """
    Manages obstacle placement and retrieval in the ocean grid.
    
    Attributes:
        obstacles (List[Obstacle]): List of all placed obstacles
        grid_width (int): Width of the grid
        grid_height (int): Height of the grid
    """
    
    def __init__(self, grid_width: int = 100, grid_height: int = 100):
        """
        Initialize the obstacle manager.
        
        Args:
            grid_width: Width of the grid in cells
            grid_height: Height of the grid in cells
        """
        self.obstacles: List[Obstacle] = []
        self.grid_width = grid_width
        self.grid_height = grid_height
    
    def add_obstacle(self, x: int, y: int, width: int, height: int, name: str = "Rock") -> None:
        """
        Add a rectangular obstacle.
        
        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Obstacle width in cells
            height: Obstacle height in cells
            name: Name/type of obstacle
        """
        obstacle = Obstacle(x, y, width, height, name)
        self.obstacles.append(obstacle)
    
    def add_random_obstacles(self, count: int, min_size: int = 5, max_size: int = 20) -> None:
        """
        Add random obstacles to the grid.
        
        Args:
            count: Number of obstacles to add
            min_size: Minimum obstacle dimension
            max_size: Maximum obstacle dimension
        """
        for i in range(count):
            width = random.randint(min_size, max_size)
            height = random.randint(min_size, max_size)
            x = random.randint(0, max(0, self.grid_width - width))
            y = random.randint(0, max(0, self.grid_height - height))
            self.add_obstacle(x, y, width, height, f"Rock_{i+1}")
    
    def get_obstacles(self) -> List[Obstacle]:
        """
        Retrieve all obstacles.
        
        Returns:
            List of all obstacles
        """
        return self.obstacles
    
    def clear_obstacles(self) -> None:
        """Remove all obstacles."""
        self.obstacles.clear()
    
    def is_collision(self, x: int, y: int) -> bool:
        """
        Check if a point collides with any obstacle.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if collision detected, False otherwise
        """
        for obstacle in self.obstacles:
            if (obstacle.x <= x < obstacle.x + obstacle.width and
                obstacle.y <= y < obstacle.y + obstacle.height):
                return True
        return False


if __name__ == "__main__":
    # Test obstacle manager
    manager = ObstacleManager(100, 100)
    manager.add_obstacle(20, 20, 30, 10, "Rock Formation")
    manager.add_obstacle(60, 60, 15, 25, "Wreck")
    manager.add_random_obstacles(5)
    
    print(f"Total obstacles: {len(manager.get_obstacles())}")
    for obs in manager.get_obstacles():
        print(f"  {obs.name}: ({obs.x}, {obs.y}) - {obs.width}x{obs.height}")
    print(f"Collision at (25, 25): {manager.is_collision(25, 25)}")
    print(f"Collision at (10, 10): {manager.is_collision(10, 10)}")
