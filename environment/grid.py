"""
Ocean Grid Environment Module

Defines the grid-based ocean map for AUV path planning simulation.
Handles grid initialization, cell states, and coordinate transformations.
"""

import numpy as np
from typing import Tuple, List


class OceanGrid:
    """
    Represents a 2D grid-based ocean environment.
    
    Attributes:
        width (int): Grid width in cells
        height (int): Grid height in cells
        cell_size (int): Physical size of each cell in pixels
        grid (np.ndarray): 2D array representing grid state (0=free, 1=obstacle)
    """
    
    def __init__(self, width: int = 100, height: int = 100, cell_size: int = 5):
        """
        Initialize the ocean grid.
        
        Args:
            width: Grid width in cells
            height: Grid height in cells
            cell_size: Pixel size of each cell for rendering
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = np.zeros((height, width), dtype=np.int32)
    
    def add_obstacle(self, x: int, y: int, width: int, height: int) -> None:
        """
        Add a rectangular obstacle to the grid.
        
        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Obstacle width in cells
            height: Obstacle height in cells
        """
        x_end = min(x + width, self.width)
        y_end = min(y + height, self.height)
        self.grid[y:y_end, x:x_end] = 1
    
    def is_free(self, x: int, y: int) -> bool:
        """
        Check if a cell is free (not an obstacle).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if cell is free, False otherwise
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] == 0
        return False
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get valid neighboring cells (8-directional movement).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            List of valid neighbor coordinates
        """
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.is_free(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors
    
    def reset(self) -> None:
        """Clear all obstacles from the grid."""
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)


if __name__ == "__main__":
    # Test grid creation
    grid = OceanGrid(100, 100, 5)
    grid.add_obstacle(20, 20, 30, 10)
    print(f"Grid initialized: {grid.width}x{grid.height}")
    print(f"Free cell at (50, 50): {grid.is_free(50, 50)}")
    print(f"Obstacle cell at (25, 25): {grid.is_free(25, 25)}")
