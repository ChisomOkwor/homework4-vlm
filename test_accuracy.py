#!/usr/bin/env python3

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image
import os
from tqdm import tqdm
import random
from homework.data import VQADataset
from homework.base_vlm import BaseVLM

def test_accuracy():
    """Test model accuracy using the trained model"""
    print("Loading test data...")
    
    # Load validation data
    test_dataset = VQADataset("valid_grader")
    print(f"Loaded {len(test_dataset)} test samples")
    
    if len(test_dataset) == 0:
        print("No test data loaded.")
        return
    
    # Load the trained model
    print("Loading trained model...")
    try:
        model = BaseVLM.load("homework/vlm_sft")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test on a subset for speed
    test_indices = random.sample(range(len(test_dataset)), min(50, len(test_dataset)))
    
    correct = 0
    total = 0
    
    print(f"Testing on {len(test_indices)} samples...")
    
    for idx in tqdm(test_indices):
        try:
            item = test_dataset[idx]
            question = item["question"]
            correct_answer = item["answer"]
            image_path = item["image_path"]
            
            # Generate model answer
            model_answer = model.answer_question(image_path, question)
            
            # Simple exact match (case insensitive)
            if model_answer.lower().strip() == correct_answer.lower().strip():
                correct += 1
                print(f"✓ Q: {question}")
                print(f"  A: {model_answer} (correct: {correct_answer})")
            else:
                print(f"✗ Q: {question}")
                print(f"  A: {model_answer} (correct: {correct_answer})")
            
            total += 1
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy

if __name__ == "__main__":
    test_accuracy() 