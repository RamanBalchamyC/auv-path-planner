"""
API Test Script

Tests all endpoints of the AUV Path Planner API
Run with: python test_api.py
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


def test_health() -> bool:
    """Test health endpoint"""
    print("Testing /health...")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {resp.json()}\n")
        return resp.status_code == 200
    except Exception as e:
        print(f"  ERROR: {e}\n")
        return False


def test_obstacles() -> bool:
    """Test obstacles endpoint"""
    print("Testing /obstacles...")
    try:
        resp = requests.get(f"{BASE_URL}/obstacles")
        data = resp.json()
        print(f"  Status: {resp.status_code}")
        print(f"  Count: {data.get('count', 0)} obstacles")
        print(f"  Response: {json.dumps(data, indent=2)}\n")
        return resp.status_code == 200
    except Exception as e:
        print(f"  ERROR: {e}\n")
        return False


def test_plan_path(start_x: int, start_y: int, goal_x: int, goal_y: int) -> bool:
    """Test path planning endpoint"""
    print(f"Testing /plan: ({start_x},{start_y}) -> ({goal_x},{goal_y})...")
    
    payload = {
        "start_x": start_x,
        "start_y": start_y,
        "goal_x": goal_x,
        "goal_y": goal_y,
        "algorithm": "astar"
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/plan", json=payload)
        data = resp.json()
        print(f"  Status: {resp.status_code}")
        print(f"  Success: {data.get('success', False)}")
        if data.get('path'):
            print(f"  Path length: {len(data['path'])} waypoints")
            print(f"  Path: {data['path'][:3]}...{data['path'][-3:] if len(data['path']) > 6 else ''}")
        print(f"  Message: {data.get('message', '')}\n")
        return resp.status_code == 200
    except Exception as e:
        print(f"  ERROR: {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("AUV PATH PLANNER - API TESTS")
    print("="*60)
    print("\nMake sure the API server is running:")
    print("  c:/python313/python.exe -m uvicorn api.main:app --port 8000\n")
    
    results = []
    
    # Test endpoints
    results.append(("Health Check", test_health()))
    results.append(("Obstacles List", test_obstacles()))
    results.append(("Path Planning 1", test_plan_path(10, 10, 90, 90)))
    results.append(("Path Planning 2", test_plan_path(5, 5, 50, 50)))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} | {test_name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)
