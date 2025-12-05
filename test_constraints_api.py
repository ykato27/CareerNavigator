"""
Test script for Constraints API
"""
import requests
import json

BASE_URL = "http://localhost:8000"
SESSION_ID = "test_session"

def test_get_constraints():
    """Test GET /api/constraints/{session_id}"""
    url = f"{BASE_URL}/api/constraints/{SESSION_ID}"
    response = requests.get(url)
    print(f"GET {url}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()
    return response.json()

def test_add_constraint():
    """Test POST /api/constraints/{session_id}"""
    url = f"{BASE_URL}/api/constraints/{SESSION_ID}"
    data = {
        "from_skill": "プログラミング基礎",
        "to_skill": "データベース設計",
        "constraint_type": "required",
        "value": 0.5
    }
    response = requests.post(url, json=data)
    print(f"POST {url}")
    print(f"Payload: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()
    return response.json()

def test_add_forbidden_constraint():
    """Test adding a forbidden constraint"""
    url = f"{BASE_URL}/api/constraints/{SESSION_ID}"
    data = {
        "from_skill": "機械学習",
        "to_skill": "プログラミング基礎",
        "constraint_type": "forbidden"
    }
    response = requests.post(url, json=data)
    print(f"POST {url}")
    print(f"Payload: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()
    return response.json()

if __name__ == "__main__":
    print("=" * 60)
    print("Constraints API Test")
    print("=" * 60)
    print()
    
    # Test 1: Get empty constraints
    print("Test 1: Get empty constraints")
    test_get_constraints()
    
    # Test 2: Add required constraint
    print("Test 2: Add required constraint")
    constraint1 = test_add_constraint()
    
    # Test 3: Add forbidden constraint
    print("Test 3: Add forbidden constraint")
    constraint2 = test_add_forbidden_constraint()
    
    # Test 4: Get all constraints
    print("Test 4: Get all constraints")
    result = test_get_constraints()
    print(f"Total constraints: {result['count']}")
    
    print("=" * 60)
    print("Tests completed!")
    print("=" * 60)
