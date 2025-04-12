#!/usr/bin/env python3
"""
Setup script for DeepCite - Academic Paper Search and Citation Tool
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is at least 3.6."""
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or higher is required.")
        sys.exit(1)

def install_requirements():
    """Install required packages from requirements.txt."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed required packages.")
    except subprocess.CalledProcessError:
        print("Error: Failed to install required packages.")
        sys.exit(1)

def make_executable():
    """Make the main script executable."""
    try:
        os.chmod("deepcite.py", 0o755)
        print("Made deepcite.py executable.")
    except Exception as e:
        print(f"Warning: Could not make deepcite.py executable: {str(e)}")

def main():
    """Main setup function."""
    print("Setting up DeepCite...")
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    print("Installing required packages...")
    install_requirements()
    
    # Make the main script executable
    make_executable()
    
    print("\nSetup complete! You can now run DeepCite with:")
    print("  python deepcite.py --interactive")
    print("  or simply: ./deepcite.py --interactive (on Unix-based systems)")

if __name__ == "__main__":
    main()