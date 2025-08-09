#!/usr/bin/env python3
"""
Script to test the accuracy of checkpoint-1283 of the CLIP model.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Test the accuracy of checkpoint-1283 of the CLIP model."""
    
    # Path to the latest checkpoint (relative to current directory)
    checkpoint_path = "clip/checkpoint-4700"
    
    print(f"Testing CLIP model accuracy for checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    try:
        # Use the fire CLI interface to run the test
        cmd = [
            sys.executable, "-m", "homework.clip", 
            "test", 
            checkpoint_path, 
            "valid_grader"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        print("-" * 60)
        
        # Run the test
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("Test completed successfully!")
            print("=" * 60)
        else:
            print(f"\nTest failed with return code: {result.returncode}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error testing the model: {e}")
        return False

if __name__ == "__main__":
    main() 