"""
PyGame Visualization Module

Renders the ocean grid, obstacles, and AUV in real-time using PyGame.
Provides visual feedback during path planning simulation.
"""

import pygame
from typing import Tuple, List, Optional
from environment.grid import OceanGrid
from environment.obstacles import Obstacle


class OceanRenderer:
    """
    Handles PyGame rendering of the ocean simulation.
    
    Attributes:
        grid (OceanGrid): The ocean grid to render
        screen (pygame.Surface): PyGame display surface
        width (int): Screen width in pixels
        height (int): Screen height in pixels
    """
    
    # Color constants
    COLOR_OCEAN = (25, 118, 210)      # Dark blue ocean
    COLOR_SAND = (238, 203, 127)      # Sandy floor
    COLOR_OBSTACLE = (139, 69, 19)    # Brown rocks
    COLOR_AUV = (76, 175, 80)         # Green AUV
    COLOR_GOAL = (255, 193, 7)        # Gold goal
    COLOR_PATH = (0, 255, 255)        # Cyan path
    COLOR_GRID = (50, 100, 150)       # Grid lines
    
    def __init__(self, grid: OceanGrid, width: int = 800, height: int = 600):
        """
        Initialize the renderer.
        
        Args:
            grid: OceanGrid instance to render
            width: Screen width in pixels
            height: Screen height in pixels
        """
        self.grid = grid
        self.width = width
        self.height = height
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AUV Path Planner - Ocean Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
    
    def draw_grid(self, show_lines: bool = False) -> None:
        """
        Draw the ocean background and grid.
        
        Args:
            show_lines: If True, draw grid lines
        """
        self.screen.fill(self.COLOR_OCEAN)
        
        if show_lines:
            cell_size = self.grid.cell_size
            for x in range(0, self.width, cell_size):
                pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.height), 1)
            for y in range(0, self.height, cell_size):
                pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.width, y), 1)
    
    def draw_obstacles(self, obstacles: List[Obstacle]) -> None:
        """
        Draw obstacles on the screen.
        
        Args:
            obstacles: List of obstacles to draw
        """
        cell_size = self.grid.cell_size
        for obstacle in obstacles:
            rect = pygame.Rect(
                obstacle.x * cell_size,
                obstacle.y * cell_size,
                obstacle.width * cell_size,
                obstacle.height * cell_size
            )
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
            pygame.draw.rect(self.screen, (100, 50, 0), rect, 2)  # Border
    
    def draw_auv(self, x: int, y: int, radius: int = 5) -> None:
        """
        Draw the AUV at the given position.
        
        Args:
            x: X coordinate (in grid cells)
            y: Y coordinate (in grid cells)
            radius: Radius of AUV circle in pixels
        """
        pixel_x = x * self.grid.cell_size + self.grid.cell_size // 2
        pixel_y = y * self.grid.cell_size + self.grid.cell_size // 2
        pygame.draw.circle(self.screen, self.COLOR_AUV, (pixel_x, pixel_y), radius)
    
    def draw_goal(self, x: int, y: int, radius: int = 8) -> None:
        """
        Draw the goal position.
        
        Args:
            x: X coordinate (in grid cells)
            y: Y coordinate (in grid cells)
            radius: Radius of goal marker in pixels
        """
        pixel_x = x * self.grid.cell_size + self.grid.cell_size // 2
        pixel_y = y * self.grid.cell_size + self.grid.cell_size // 2
        pygame.draw.circle(self.screen, self.COLOR_GOAL, (pixel_x, pixel_y), radius)
        pygame.draw.circle(self.screen, (255, 255, 0), (pixel_x, pixel_y), radius, 2)
    
    def draw_path(self, path: List[Tuple[int, int]]) -> None:
        """
        Draw the planned path.
        
        Args:
            path: List of coordinates representing the path
        """
        if not path or len(path) < 2:
            return
        
        cell_size = self.grid.cell_size
        
        # Draw line segments between consecutive path points
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            px1 = int(x1 * cell_size + cell_size // 2)
            py1 = int(y1 * cell_size + cell_size // 2)
            px2 = int(x2 * cell_size + cell_size // 2)
            py2 = int(y2 * cell_size + cell_size // 2)
            pygame.draw.line(self.screen, self.COLOR_PATH, (px1, py1), (px2, py2), 2)
    
    def draw_text(self, text: str, x: int, y: int, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        """
        Draw text on screen.
        
        Args:
            text: Text to display
            x: X position in pixels
            y: Y position in pixels
            color: RGB color tuple
        """
        surface = self.font.render(text, True, color)
        self.screen.blit(surface, (x, y))
    
    def update_display(self) -> None:
        """Update the PyGame display."""
        pygame.display.flip()
    
    def set_fps(self, fps: int = 60) -> None:
        """
        Set frames per second limit.
        
        Args:
            fps: Frames per second
        """
        self.clock.tick(fps)
    
    def handle_events(self) -> bool:
        """
        Handle PyGame events.
        
        Returns:
            False if window closed, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def close(self) -> None:
        """Close the PyGame window."""
        pygame.quit()


if __name__ == "__main__":
    # Test renderer
    from environment.obstacles import ObstacleManager
    
    grid = OceanGrid(100, 100, 5)
    renderer = OceanRenderer(grid, 800, 600)
    
    manager = ObstacleManager(100, 100)
    manager.add_obstacle(20, 20, 30, 10)
    manager.add_obstacle(60, 60, 15, 25)
    
    # Simple render loop
    running = True
    frame = 0
    while running:
        renderer.draw_grid(show_lines=False)
        renderer.draw_obstacles(manager.get_obstacles())
        renderer.draw_auv(50, 50)
        renderer.draw_goal(80, 80)
        renderer.draw_text(f"Frame: {frame}", 10, 10)
        renderer.update_display()
        running = renderer.handle_events()
        renderer.set_fps(30)
        frame += 1
    
    renderer.close()
