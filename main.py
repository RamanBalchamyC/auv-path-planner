"""
AUV Path Planner - Main Simulation

Supports:
1. A* Pathfinding (Classical)
2. Q-Learning (Reinforcement Learning)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.grid import OceanGrid
from environment.obstacles import ObstacleManager
from environment.renderer import OceanRenderer
from ai.astar import AStarPlanner
from ai.qlearning import QLearningAgent


def run_astar():
    """A* Pathfinding Demo"""
    print("\n" + "="*60)
    print("A* PATHFINDING DEMO")
    print("="*60)
    
    # Setup
    grid = OceanGrid(100, 100, 5)
    obstacles = ObstacleManager(100, 100)
    obstacles.add_obstacle(20, 20, 30, 10)
    obstacles.add_obstacle(60, 60, 20, 30)
    obstacles.add_random_obstacles(8)
    
    for obs in obstacles.get_obstacles():
        grid.add_obstacle(obs.x, obs.y, obs.width, obs.height)
    
    renderer = OceanRenderer(grid, 800, 600)
    planner = AStarPlanner(grid)
    
    start, goal = (10, 10), (90, 90)
    path = planner.plan(start, goal)
    
    print(f"Obstacles: {len(obstacles.get_obstacles())}")
    print(f"Path: {len(path) if path else 'NOT FOUND'} waypoints")
    print("Press ESC to exit\n")
    
    # Run
    auv_pos = start
    idx = 0
    running = True
    
    while running:
        running = renderer.handle_events()
        renderer.draw_grid()
        renderer.draw_obstacles(obstacles.get_obstacles())
        
        if path:
            renderer.draw_path(path)
            if idx < len(path) - 1:
                idx += 1
                auv_pos = path[idx]
        
        renderer.draw_auv(auv_pos[0], auv_pos[1])
        renderer.draw_goal(goal[0], goal[1])
        renderer.draw_text(f"A* | Step {idx}/{len(path) if path else 0}", 10, 10)
        renderer.update_display()
        renderer.set_fps(30)
    
    renderer.close()
    print("Demo ended.\n")


def run_qlearning():
    """Q-Learning Training Demo"""
    print("\n" + "="*60)
    print("Q-LEARNING TRAINING DEMO")
    print("="*60)
    
    # Setup
    grid = OceanGrid(50, 50, 5)
    grid.add_obstacle(10, 10, 20, 10)
    grid.add_obstacle(30, 25, 15, 20)
    
    renderer = OceanRenderer(grid, 800, 600)
    agent = QLearningAgent(grid, learning_rate=0.1, epsilon=0.2)
    
    start, goal = (5, 5), (45, 45)
    episodes = 50
    
    # Train
    print(f"\nTraining for {episodes} episodes...")
    best = float('-inf')
    
    for ep in range(episodes):
        reward, steps, path = agent.learn(start, goal, max_steps=150)
        best = max(best, reward)
        if ep % 10 == 0:
            print(f"  Episode {ep:2d}: Reward={reward:6.0f} (Best: {best:6.0f})")
    
    print(f"\nTraining done! Best reward: {best:.0f}")
    print("Visualizing learned path...\n")
    
    # Get final path
    _, _, final_path = agent.learn(start, goal, max_steps=150)
    
    # Visualize
    auv_pos = start
    idx = 0
    running = True
    
    while running:
        running = renderer.handle_events()
        renderer.draw_grid()
        
        if final_path:
            renderer.draw_path(final_path[:min(idx + 1, len(final_path))])
            if idx < len(final_path) - 1:
                idx += 1
                auv_pos = final_path[idx]
        
        renderer.draw_auv(auv_pos[0], auv_pos[1])
        renderer.draw_goal(goal[0], goal[1])
        renderer.draw_text(f"Q-Learning | Path: {len(final_path)} steps", 10, 10)
        renderer.update_display()
        renderer.set_fps(20)
    
    renderer.close()
    print("Demo ended.\n")


def main():
    """Main Menu"""
    print("\n" + "="*60)
    print("AUV PATH PLANNER")
    print("="*60)
    print("\n1. A* Pathfinding")
    print("2. Q-Learning Agent")
    print("\nSelect (1-2): ", end="")
    
    choice = input().strip()
    
    if choice == "1":
        run_astar()
    elif choice == "2":
        run_qlearning()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
