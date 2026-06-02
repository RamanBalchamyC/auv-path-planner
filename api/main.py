"""
FastAPI Server Module

Exposes the AUV path planner as a REST API.
Endpoints for path planning using different algorithms.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.grid import OceanGrid
from environment.obstacles import ObstacleManager
from ai.astar import AStarPlanner


class PathRequest(BaseModel):
    """Request model for path planning."""
    start_x: int
    start_y: int
    goal_x: int
    goal_y: int
    algorithm: str = "astar"  # "astar", "qlearning", "dqn"


class PathResponse(BaseModel):
    """Response model for path planning."""
    success: bool
    path: List[Tuple[int, int]] = []
    path_length: int = 0
    message: str = ""


app = FastAPI(
    title="AUV Path Planner API",
    description="REST API for autonomous underwater vehicle path planning",
    version="1.0.0"
)

# Global instances
grid = OceanGrid(100, 100, 5)
obstacle_manager = ObstacleManager(100, 100)
astar_planner = AStarPlanner(grid)

# Initialize some obstacles
obstacle_manager.add_obstacle(20, 20, 30, 10, "Rock Formation")
obstacle_manager.add_obstacle(60, 60, 20, 30, "Wreck")
for obs in obstacle_manager.get_obstacles():
    grid.add_obstacle(obs.x, obs.y, obs.width, obs.height)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AUV Path Planner API",
        "version": "1.0.0",
        "endpoints": [
            "GET /health",
            "POST /plan",
            "GET /obstacles"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "grid_size": f"{grid.width}x{grid.height}"}


@app.get("/obstacles")
async def get_obstacles():
    """Get all obstacles on the grid."""
    obs_list = []
    for obs in obstacle_manager.get_obstacles():
        obs_list.append({
            "name": obs.name,
            "x": obs.x,
            "y": obs.y,
            "width": obs.width,
            "height": obs.height
        })
    return {"obstacles": obs_list, "count": len(obs_list)}


@app.post("/plan", response_model=PathResponse)
async def plan_path(request: PathRequest) -> PathResponse:
    """
    Plan a path from start to goal.
    
    Args:
        request: PathRequest with start, goal, and algorithm
        
    Returns:
        PathResponse with path or error message
    """
    # Validate coordinates
    if not (0 <= request.start_x < grid.width and 0 <= request.start_y < grid.height):
        raise HTTPException(status_code=400, detail="Start position out of bounds")
    if not (0 <= request.goal_x < grid.width and 0 <= request.goal_y < grid.height):
        raise HTTPException(status_code=400, detail="Goal position out of bounds")
    
    # Check if start/goal are free
    if not grid.is_free(request.start_x, request.start_y):
        raise HTTPException(status_code=400, detail="Start position is an obstacle")
    if not grid.is_free(request.goal_x, request.goal_y):
        raise HTTPException(status_code=400, detail="Goal position is an obstacle")
    
    start = (request.start_x, request.start_y)
    goal = (request.goal_x, request.goal_y)
    
    try:
        if request.algorithm.lower() == "astar":
            path = astar_planner.plan(start, goal)
            if path:
                return PathResponse(
                    success=True,
                    path=path,
                    path_length=len(path),
                    message=f"Path found with {len(path)} waypoints"
                )
            else:
                return PathResponse(
                    success=False,
                    message="No path found from start to goal"
                )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown algorithm: {request.algorithm}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting AUV Path Planner API...")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
