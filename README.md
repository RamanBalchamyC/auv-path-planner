# AUV Path Planner

Autonomous Underwater Vehicle (AUV) path planning system with simulation and AI-driven navigation.

## 🌊 Project Overview

A comprehensive AUV path planner that combines:

- **PyGame Simulation**: Real-time 2D ocean environment visualization
- **Path Planning Algorithms**: A\* (classical), Q-Learning (RL), DQN (deep learning)
- **REST API**: FastAPI server for production deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Layer 3: API Layer              │
│      (FastAPI REST Server)              │
└──────────────┬──────────────────────────┘
               │
┌──────────────v──────────────────────────┐
│       Layer 2: AI Brain                  │
│  A* | Q-Learning | DQN Algorithms       │
└──────────────┬──────────────────────────┘
               │
┌──────────────v──────────────────────────┐
│   Layer 1: Simulation Environment       │
│  PyGame | Grid | Obstacles | Renderer   │
└─────────────────────────────────────────┘
```

## 📁 Project Structure

```
auv-path-planner/
│
├── environment/
│   ├── grid.py          # Ocean grid map implementation
│   ├── obstacles.py     # Obstacle placement and management
│   └── renderer.py      # PyGame visualization engine
│
├── ai/
│   ├── astar.py         # A* pathfinding algorithm
│   ├── qlearning.py     # Q-Learning reinforcement learning agent
│   └── dqn.py           # Deep Q-Network (bonus feature)
│
├── api/
│   └── main.py          # FastAPI server with REST endpoints
│
├── main.py              # Simulation launcher (A* & Q-Learning)
├── run.py               # Quick launcher menu
├── test_api.py          # API endpoint tests
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/auv-path-planner.git
cd auv-path-planner

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start (Easy Way)

```bash
# Run the launcher menu
python run.py
```

Choose from:

1. **Run Simulation** - PyGame visualization of A\* or Q-Learning
2. **Start API Server** - FastAPI on `http://localhost:8000`
3. **Test API** - Automated endpoint tests

### Manual Execution

**Run Simulation**

```bash
c:/python313/python.exe main.py
```

Select 1 for A\* demo or 2 for Q-Learning demo

**Start API Server**

```bash
c:/python313/python.exe -m uvicorn api.main:app --port 8000
```

Then visit: `http://localhost:8000/docs` for interactive API docs

**Test API (requires running server)**

```bash
c:/python313/python.exe test_api.py
```

#### API Endpoints

- **GET `/`** - API information
- **GET `/health`** - Health check
- **GET `/obstacles`** - List all obstacles
- **POST `/plan`** - Plan a path
  ```json
  {
    "start_x": 10,
    "start_y": 10,
    "goal_x": 90,
    "goal_y": 90,
    "algorithm": "astar"
  }
  ```

## 🧠 Algorithms

### 1. **A\* Pathfinding** ⭐ (Week 1 - Foundation)

- Classical deterministic pathfinding
- Optimal shortest path
- Uses heuristic (Euclidean distance)
- Fast and reliable

**Files**: [`ai/astar.py`](ai/astar.py)

```python
from environment.grid import OceanGrid
from ai.astar import AStarPlanner

grid = OceanGrid(100, 100)
planner = AStarPlanner(grid)
path = planner.plan((10, 10), (90, 90))
```

### 2. **Q-Learning** 🤖 (Week 1 - Add-on)

- Reinforcement learning agent
- Learns optimal policy through trial and error
- Rewards: +100 goal, -50 collision, -1 per step
- Suitable for dynamic environments

**Files**: [`ai/qlearning.py`](ai/qlearning.py)

```python
from ai.qlearning import QLearningAgent

agent = QLearningAgent(grid, learning_rate=0.1)
for episode in range(100):
    reward, steps, path = agent.learn((10, 10), (90, 90))
```

### 3. **Deep Q-Network (DQN)** 🧠 (Bonus - Advanced)

- Neural network-based Q-learning
- Experience replay buffer
- Target network for stable learning
- State-of-the-art deep RL

**Files**: [`ai/dqn.py`](ai/dqn.py)

```python
from ai.dqn import DQNAgent

agent = DQNAgent(grid, learning_rate=0.001)
for episode in range(50):
    reward, steps = agent.learn((10, 10), (90, 90))
    if episode % 10 == 9:
        agent.update_target_network()
```

## 📊 Tech Stack

| Component         | Technology      | Purpose                        |
| ----------------- | --------------- | ------------------------------ |
| **Core Language** | Python 3.8+     | High-level development         |
| **Simulation**    | PyGame 2.5+     | 2D graphics & visualization    |
| **Numerics**      | NumPy 1.24+     | Grid operations & matrices     |
| **Deep Learning** | PyTorch 2.1+    | DQN neural networks (optional) |
| **Web Framework** | FastAPI 0.104+  | REST API server                |
| **Server**        | Uvicorn 0.24+   | ASGI application server        |
| **Dashboard**     | Streamlit 1.28+ | Monitoring UI (optional)       |

## 🎓 Development Roadmap

### Week 1

- **Day 1-2**: Simulation layer (grid, obstacles, renderer)
- **Day 3**: A\* pathfinding
- **Day 4**: Q-Learning agent
- **Day 5**: API & documentation

### Week 2+ (Future)

- DQN implementation & training
- Advanced visualization with Streamlit
- Real sensor simulation (noise, localization)
- Physics simulation (hydrodynamics, buoyancy)
- GitHub CI/CD pipeline

## 🔧 Configuration

Edit simulation parameters in `main.py`:

```python
GRID_WIDTH = 100        # Grid width in cells
GRID_HEIGHT = 100       # Grid height in cells
CELL_SIZE = 5           # Pixel size per cell
SCREEN_WIDTH = 800      # Display width
SCREEN_HEIGHT = 600     # Display height
```

## 📈 Performance

| Algorithm      | Speed  | Optimality | Learning |
| -------------- | ------ | ---------- | -------- |
| **A\***        | Fast   | Optimal    | No       |
| **Q-Learning** | Medium | Good       | Yes      |
| **DQN**        | Slow   | Good       | Yes      |

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- Advanced obstacle shapes (circles, polygons)
- Multi-agent pathfinding
- Real sensor simulation
- 3D ocean environment
- Continuous path smoothing

## 📝 License

MIT License - Feel free to use this in your projects!

## 🔗 Resources

- **A\* Algorithm**: [Wikipedia](https://en.wikipedia.org/wiki/A*_search_algorithm)
- **Q-Learning**: [OpenAI Spinning Up](https://spinningup.openai.com/)
- **DQN Paper**: [Nature 2015](https://www.nature.com/articles/nature14236)
- **PyGame Docs**: [pygame.org](https://www.pygame.org/)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)

## 👨‍💻 Author

Built as a portfolio project for marine robotics career transition.

---

**Happy path planning! 🚀**
