"""
A* Pathfinding Algorithm Module

Implements the A* algorithm for optimal pathfinding on a grid.
Used as the foundation for deterministic path planning.
"""

import heapq
from typing import List, Tuple, Optional
from environment.grid import OceanGrid
import math


class AStarPlanner:
    """
    A* path planning algorithm.
    
    Attributes:
        grid (OceanGrid): The ocean grid
        allow_diagonal (bool): Whether diagonal movement is allowed
    """
    
    def __init__(self, grid: OceanGrid, allow_diagonal: bool = True):
        """
        Initialize A* planner.
        
        Args:
            grid: OceanGrid instance
            allow_diagonal: Allow 8-directional movement if True, 4-directional if False
        """
        self.grid = grid
        self.allow_diagonal = allow_diagonal
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Calculate heuristic distance (Euclidean).
        
        Args:
            pos: Current position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            Heuristic distance estimate
        """
        dx = pos[0] - goal[0]
        dy = pos[1] - goal[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighbors for a position.
        
        Args:
            pos: Current position (x, y)
            
        Returns:
            List of valid neighbor positions
        """
        x, y = pos
        neighbors = []
        
        # 4-directional neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.grid.is_free(nx, ny):
                neighbors.append((nx, ny))
        
        # Diagonal neighbors (if enabled)
        if self.allow_diagonal:
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if self.grid.is_free(nx, ny):
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find optimal path from start to goal using A* algorithm.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of coordinates forming the path, or None if no path exists
        """
        if not self.grid.is_free(start[0], start[1]):
            return None
        if not self.grid.is_free(goal[0], goal[1]):
            return None
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        closed_set = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                node = goal
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                path.reverse()
                return path
            
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + self.heuristic(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found


if __name__ == "__main__":
    # Test A* planner
    from environment.obstacles import ObstacleManager
    
    grid = OceanGrid(100, 100, 5)
    manager = ObstacleManager(100, 100)
    manager.add_obstacle(20, 20, 30, 20)
    manager.add_obstacle(60, 30, 20, 40)
    
    for obs in manager.get_obstacles():
        grid.add_obstacle(obs.x, obs.y, obs.width, obs.height)
    
    planner = AStarPlanner(grid)
    path = planner.plan((5, 5), (95, 95))
    
    if path:
        print(f"Path found! Length: {len(path)} cells")
        print(f"Start: {path[0]}, End: {path[-1]}")
    else:
        print("No path found!")
