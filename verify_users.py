#!/usr/bin/env python3
"""
Script to verify users.json and test password authentication
"""

import json
import bcrypt
from pathlib import Path

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a bcrypt hash"""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        print(f"Error verifying password: {e}")
        return False

def test_users():
    """Test users.json file and password verification"""
    
    script_dir = Path(__file__).parent
    users_file = script_dir / "users.json"
    
    print(f"Checking users file: {users_file}")
    print()
    
    if not users_file.exists():
        print("✗ users.json file does NOT exist!")
        print("  Run: python3 create_user.py --all")
        return
    
    # Load users
    try:
        with open(users_file, 'r') as f:
            users_data = json.load(f)
    except Exception as e:
        print(f"✗ Error loading users.json: {e}")
        return
    
    users = users_data.get("users", [])
    
    if not users:
        print("✗ No users found in users.json!")
        print("  Run: python3 create_user.py --all")
        return
    
    print(f"✓ Found {len(users)} user(s) in users.json")
    print()
    
    # Test each user
    test_passwords = {
        "user1": "vYC3flYH",
        "user2": "lyid9fhb",
        "user3": "zH3hiiZI",
    }
    
    print("Testing password verification:")
    print("-" * 50)
    
    for user in users:
        username = user.get("username")
        password_hash = user.get("password_hash")
        
        if not username:
            print(f"✗ User missing username: {user}")
            continue
        
        if not password_hash:
            print(f"✗ User '{username}' missing password_hash")
            continue
        
        # Test password
        if username in test_passwords:
            test_password = test_passwords[username]
            is_valid = verify_password(test_password, password_hash)
            
            if is_valid:
                print(f"✓ {username}: Password verification PASSED")
            else:
                print(f"✗ {username}: Password verification FAILED")
                print(f"    Expected password: {test_password}")
                print(f"    Hash: {password_hash[:30]}...")
        else:
            print(f"ℹ {username}: No test password defined")
    
    print()
    print("Users in file:")
    for user in users:
        username = user.get("username", "unknown")
        password_hash = user.get("password_hash", "missing")
        print(f"  - {username}: {password_hash[:30]}...")
    
    print()
    print("To fix issues:")
    print("  python3 create_user.py --all")
    print("  sudo systemctl restart rag-anything-backend")

if __name__ == "__main__":
    test_users()

