#!/usr/bin/env python3
"""
Detailed script to test the accuracy of checkpoint-1283 of the CLIP model.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

def main():
    """Test the accuracy of checkpoint-1283 of the CLIP model with detailed results."""
    
    # Path to the checkpoint-1283 (using the path that worked in the first test)
    checkpoint_path = "clip/checkpoint-1283"
    
    print("=" * 80)
    print("CLIP MODEL ACCURACY TEST - CHECKPOINT-1283")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test Dataset: valid_grader")
    print("-" * 80)
    
    # Check if checkpoint exists (check in homework directory)
    checkpoint_dir = Path("homework") / checkpoint_path
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    print(f"✓ Checkpoint directory found: {checkpoint_dir}")
    
    # Check for required files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    for file in required_files:
        if (checkpoint_dir / file).exists():
            print(f"✓ Found required file: {file}")
        else:
            print(f"✗ Missing required file: {file}")
            return False
    
    print("-" * 80)
    print("Starting evaluation...")
    print("-" * 80)
    
    try:
        # Run the test using the fire CLI interface
        cmd = [
            sys.executable, "-m", "homework.clip", 
            "test", 
            checkpoint_path, 
            "valid_grader"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Run the test
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            # Extract accuracy from output
            accuracy = None
            for line in result.stdout.split('\n'):
                if line.startswith('Accuracy:'):
                    try:
                        accuracy = float(line.split(':')[1].strip())
                        break
                    except:
                        pass
            
            print("\n" + "=" * 80)
            print("EVALUATION RESULTS")
            print("=" * 80)
            print(f"✓ Test completed successfully!")
            if accuracy is not None:
                print(f"✓ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"✓ Correct Predictions: {int(accuracy * 200)} out of 200")
                print(f"✓ Incorrect Predictions: {200 - int(accuracy * 200)} out of 200")
            print("=" * 80)
            
            # Save results to file
            results = {
                "timestamp": datetime.now().isoformat(),
                "checkpoint": checkpoint_path,
                "dataset": "valid_grader",
                "accuracy": accuracy,
                "total_samples": 200,
                "correct_predictions": int(accuracy * 200) if accuracy else None,
                "incorrect_predictions": 200 - int(accuracy * 200) if accuracy else None
            }
            
            with open("clip_checkpoint_1283_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"✓ Results saved to: clip_checkpoint_1283_results.json")
            
            return True
        else:
            print(f"\n✗ Test failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error testing the model: {e}")
        return False

if __name__ == "__main__":
    main() 