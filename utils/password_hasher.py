"""
Password Hasher Utility
=======================
Generate bcrypt password hashes for user authentication.

Usage:
    python utils/password_hasher.py

Or in Python:
    from utils.password_hasher import hash_password
    hashed = hash_password("your_password")
"""

import bcrypt
import secrets


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode(), hashed.encode())


def generate_cookie_key(length: int = 32) -> str:
    """Generate a secure random cookie key."""
    return secrets.token_hex(length)


if __name__ == "__main__":
    print("\n=== Snoonu ML Password Hasher ===\n")

    # Generate cookie key
    print("1. Generate Cookie Key:")
    key = generate_cookie_key()
    print(f"   {key}\n")

    # Hash password
    print("2. Hash a Password:")
    password = input("   Enter password to hash: ")
    if password:
        hashed = hash_password(password)
        print(f"\n   Hashed: {hashed}")

        # Verify
        verified = verify_password(password, hashed)
        print(f"   Verified: {verified}\n")

    print("\nAdd these to your .streamlit/secrets.toml file.")
