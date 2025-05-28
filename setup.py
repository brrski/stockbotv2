# requirements.txt
#pandas==2.2.0
#torch==2.2.0
#transformers==4.37.2
#yfinance==0.2.36
#requests==2.31.0

# setup.py
import subprocess
import sys

def setup_environment():
    """Install required packages using pip"""
    packages = [
        'pandas',
        'torch',
        'transformers',
        'yfinance',
        'requests',
        'PyQt6',

    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            return False
    
    # Verify installations
    try:
        import pandas
        import torch
        import transformers
        import yfinance
        import requests
        
        print("\nAll packages installed successfully!")
        print(f"Pandas version: {pandas.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        return True
    except ImportError as e:
        print(f"Error verifying installations: {str(e)}")
        return False

if __name__ == "__main__":
    setup_environment()
