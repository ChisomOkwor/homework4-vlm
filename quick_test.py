#!/usr/bin/env python3
"""
Quick test script to evaluate VLM model accuracy on validation data.
Tests multiple checkpoints and finds the best one.
"""

import json
from pathlib import Path
from homework.finetune import load
from homework.data import VQADataset

def test_checkpoint(checkpoint_path):
    """Test a specific checkpoint and return accuracy"""
    print(f"\n{'='*60}")
    print(f"Testing checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    try:
        # Load the trained model
        print("Loading trained model...")
        model = load(checkpoint_path)
        print(f"Model loaded successfully from {checkpoint_path}!")
        
        # Load validation dataset
        print("Loading validation dataset...")
        val_dataset = VQADataset("valid_grader", Path("data"))
        print(f"Loaded {len(val_dataset)} validation samples")
        
        # Test accuracy
        correct = 0
        total = 0
        question_type_stats = {}
        
        print("Testing model accuracy...")
        for i, item in enumerate(val_dataset):
            try:
                question = item["question"]
                answer = item["answer"]
                image_path = item["image_path"]
                
                # Generate answer
                generated = model.generate(image_path, question)
                
                # Simple exact match (case-insensitive)
                is_correct = generated.strip().lower() == answer.strip().lower()
                
                if is_correct:
                    correct += 1
                
                total += 1
                
                # Track by question type
                question_type = question.split()[0].lower() if question else "unknown"
                if question_type not in question_type_stats:
                    question_type_stats[question_type] = {"correct": 0, "total": 0}
                
                if is_correct:
                    question_type_stats[question_type]["correct"] += 1
                question_type_stats[question_type]["total"] += 1
                
                # Progress indicator
                if (i + 1) % 50 == 0:
                    current_acc = (correct / total) * 100
                    print(f"Progress: {i+1}/{len(val_dataset)} - Current accuracy: {current_acc:.2f}%")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {checkpoint_path}")
        print(f"{'='*60}")
        print(f"Total samples tested: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Show breakdown by question type
        print(f"\nAccuracy by question type:")
        for qtype, stats in question_type_stats.items():
            if stats["total"] > 0:
                qtype_acc = (stats["correct"] / stats["total"]) * 100
                print(f"  {qtype}: {qtype_acc:.2f}% ({stats['correct']}/{stats['total']})")
        
        return accuracy, question_type_stats
        
    except Exception as e:
        print(f"Error testing {checkpoint_path}: {e}")
        return 0.0, {}

def main():
    """Test multiple checkpoints and find the best one"""
    checkpoints = [
        "vlm_sft/checkpoint-2700",
        "vlm_sft/checkpoint-2750", 
        "vlm_sft/checkpoint-2800",
        "vlm_sft/checkpoint-2850"
    ]
    
    results = {}
    
    print("🧪 TESTING MULTIPLE CHECKPOINTS TO FIND THE BEST ONE")
    print("="*80)
    
    for checkpoint in checkpoints:
        accuracy, stats = test_checkpoint(checkpoint)
        results[checkpoint] = {
            "accuracy": accuracy,
            "stats": stats
        }
    
    # Find the best checkpoint
    print(f"\n{'='*80}")
    print("🏆 FINAL COMPARISON")
    print(f"{'='*80}")
    
    best_checkpoint = None
    best_accuracy = 0
    
    for checkpoint, result in results.items():
        accuracy = result["accuracy"]
        print(f"{checkpoint}: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_checkpoint = checkpoint
    
    print(f"\n🎯 BEST CHECKPOINT: {best_checkpoint}")
    print(f"🏆 BEST ACCURACY: {best_accuracy:.2f}%")
    
    if best_accuracy >= 70.0:
        print(f"✅ TARGET ACHIEVED! 70%+ accuracy reached!")
    else:
        print(f"📈 Still need {(70.0 - best_accuracy):.2f}% to reach target")
    
    return best_checkpoint, best_accuracy

if __name__ == "__main__":
    main() 