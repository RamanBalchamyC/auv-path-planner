"""
Quick Run Script

Run this to launch the simulation or API server
"""

import subprocess
import sys

PYTHON = "c:/python313/python.exe"


def main():
    print("\n" + "="*60)
    print("AUV PATH PLANNER - LAUNCHER")
    print("="*60)
    print("\n1. Run Simulation (PyGame)")
    print("2. Start API Server (FastAPI)")
    print("3. Test API (requires server running)")
    print("\nSelect (1-3): ", end="")
    
    choice = input().strip()
    
    if choice == "1":
        print("\nStarting simulation...\n")
        subprocess.run([PYTHON, "main.py"])
    
    elif choice == "2":
        print("\nStarting API server on http://localhost:8000")
        print("API Docs: http://localhost:8000/docs\n")
        subprocess.run([PYTHON, "-m", "uvicorn", "api.main:app", "--port", "8000"])
    
    elif choice == "3":
        print("\nRunning API tests...\n")
        subprocess.run([PYTHON, "test_api.py"])
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited")
