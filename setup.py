#!/usr/bin/env python3
"""
Setup script for AI SQL Assistant
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def setup_virtual_environment():
    """Set up virtual environment"""
    if not Path("venv").exists():
        if not run_command("python3 -m venv venv", "Creating virtual environment"):
            return False
    
    # Activate virtual environment and install dependencies
    activate_cmd = "source venv/bin/activate" if os.name != 'nt' else "venv\\Scripts\\activate"
    pip_cmd = f"{activate_cmd} && pip install --upgrade pip"
    
    if not run_command(pip_cmd, "Upgrading pip"):
        return False
    
    install_cmd = f"{activate_cmd} && pip install -r requirements.txt"
    if not run_command(install_cmd, "Installing dependencies"):
        return False
    
    return True

def setup_environment_file():
    """Set up environment configuration"""
    if not Path(".env").exists():
        if Path(".env.example").exists():
            shutil.copy(".env.example", ".env")
            print("📄 Created .env file from .env.example")
            print("⚠️  Please update .env with your actual configuration values")
        else:
            print("❌ .env.example file not found")
            return False
    else:
        print("✅ .env file already exists")
    return True

def run_tests():
    """Run the test suite"""
    activate_cmd = "source venv/bin/activate" if os.name != 'nt' else "venv\\Scripts\\activate"
    test_cmd = f"{activate_cmd} && python -m pytest test_suite.py -v"
    
    return run_command(test_cmd, "Running test suite")

def check_database_connection():
    """Check if database connection works"""
    activate_cmd = "source venv/bin/activate" if os.name != 'nt' else "venv\\Scripts\\activate"
    test_cmd = f"{activate_cmd} && python -c 'from db_runner import test_connection; print(\"✅ Database connection successful\" if test_connection() else \"❌ Database connection failed\")'"
    
    return run_command(test_cmd, "Testing database connection")

def main():
    """Main setup function"""
    print("🚀 Setting up AI SQL Assistant...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Set up virtual environment
    if not setup_virtual_environment():
        print("❌ Failed to set up virtual environment")
        sys.exit(1)
    
    # Set up environment file
    if not setup_environment_file():
        print("❌ Failed to set up environment file")
        sys.exit(1)
    
    # Run tests
    print("\n🧪 Running tests...")
    if run_tests():
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed, but setup can continue")
    
    # Test database connection (optional)
    print("\n🗄️  Testing database connection...")
    if check_database_connection():
        print("✅ Database connection successful!")
    else:
        print("⚠️  Database connection failed. Please check your .env configuration")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Update .env file with your actual configuration")
    print("2. Ensure PostgreSQL is running with Chinook database")
    print("3. Run the application:")
    print("   source venv/bin/activate && streamlit run app.py")
    print("\n💡 For development:")
    print("   - Run tests: source venv/bin/activate && pytest")
    print("   - Format code: source venv/bin/activate && black .")
    print("   - Lint code: source venv/bin/activate && flake8")

if __name__ == "__main__":
    main()
