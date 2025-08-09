#!/usr/bin/env python3
"""
Analyze CLIP model performance to identify weaknesses and improvement opportunities.
"""

import torch
import torchvision as tv
from PIL import Image
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

from homework.clip import load
from homework.data import MultiChoiceQADataset
from transformers import AutoProcessor

def analyze_model_performance():
    print("Loading CLIP model and test dataset...")
    
    # Load model and dataset
    clip = load("clip/best_checkpoint")
    clip = clip.model.to("mps" if torch.backends.mps.is_available() else "cpu")
    clip.eval()
    
    testset = MultiChoiceQADataset("valid_grader")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    
    image_processor = tv.transforms.Compose([
        tv.transforms.Resize(192),
        tv.transforms.CenterCrop(192),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Analysis containers
    results = []
    confidence_scores = []
    error_types = defaultdict(int)
    
    print(f"Analyzing {len(testset)} test cases...")
    
    with torch.no_grad():
        for i, pair in enumerate(testset):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(testset)}")
                
            # Process image
            image = Image.open(pair["image_path"]).convert("RGB")
            pixel_values = image_processor(image).unsqueeze(0).to(device)
            if device != "cpu":
                pixel_values = pixel_values.bfloat16()
            
            # Process text candidates
            text_inputs = processor(
                text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = text_inputs["input_ids"].long().to(device)
            attention_mask = text_inputs["attention_mask"].to(device)
            
            # Get model predictions
            vision_feature, text_feature, temperature = clip(pixel_values, input_ids, attention_mask)
            
            # Compute similarities and prediction
            similarities = torch.matmul(vision_feature, text_feature.T) / temperature
            prediction = similarities.argmax(dim=-1).item()
            confidence = torch.softmax(similarities, dim=-1).max().item()
            
            # Analyze result
            is_correct = prediction == pair["correct_index"]
            
            results.append({
                'image_path': str(pair["image_path"]),
                'question': pair.get("question", ""),
                'candidates': pair["candidates"],
                'correct_index': pair["correct_index"],
                'predicted_index': prediction,
                'is_correct': is_correct,
                'confidence': confidence,
                'similarities': similarities.cpu().tolist()[0],
                'correct_similarity': similarities[0, pair["correct_index"]].item(),
                'predicted_similarity': similarities[0, prediction].item(),
            })
            
            confidence_scores.append(confidence)
            
            # Categorize errors
            if not is_correct:
                # Analyze what type of error this is
                correct_sim = similarities[0, pair["correct_index"]].item()
                predicted_sim = similarities[0, prediction].item()
                margin = predicted_sim - correct_sim
                
                if margin > 0.1:
                    error_types["high_confidence_wrong"] += 1
                elif margin > 0.01:
                    error_types["low_confidence_wrong"] += 1
                else:
                    error_types["very_close_wrong"] += 1
    
    # Calculate statistics
    total_cases = len(results)
    correct_cases = sum(r['is_correct'] for r in results)
    accuracy = correct_cases / total_cases
    
    avg_confidence = np.mean(confidence_scores)
    correct_confidences = [r['confidence'] for r in results if r['is_correct']]
    incorrect_confidences = [r['confidence'] for r in results if not r['is_correct']]
    
    print("\n" + "="*60)
    print("CLIP MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    print(f"Overall Accuracy: {accuracy:.4f} ({correct_cases}/{total_cases})")
    print(f"Average Confidence: {avg_confidence:.4f}")
    
    if correct_confidences:
        print(f"Avg Confidence (Correct): {np.mean(correct_confidences):.4f}")
    if incorrect_confidences:
        print(f"Avg Confidence (Incorrect): {np.mean(incorrect_confidences):.4f}")
    
    print("\nError Analysis:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count} ({count/total_cases*100:.1f}%)")
    
    # Find most confident wrong predictions
    wrong_results = [r for r in results if not r['is_correct']]
    if wrong_results:
        wrong_results.sort(key=lambda x: x['confidence'], reverse=True)
        print(f"\nMost Confident Wrong Predictions (top 5):")
        for i, r in enumerate(wrong_results[:5]):
            print(f"  {i+1}. Confidence: {r['confidence']:.3f}")
            print(f"     Predicted: '{r['candidates'][r['predicted_index']]}'")
            print(f"     Correct: '{r['candidates'][r['correct_index']]}'")
            print(f"     Image: {Path(r['image_path']).name}")
            print()
    
    # Find least confident correct predictions
    correct_results = [r for r in results if r['is_correct']]
    if correct_results:
        correct_results.sort(key=lambda x: x['confidence'])
        print(f"Least Confident Correct Predictions (top 5):")
        for i, r in enumerate(correct_results[:5]):
            print(f"  {i+1}. Confidence: {r['confidence']:.3f}")
            print(f"     Answer: '{r['candidates'][r['correct_index']]}'")
            print(f"     Image: {Path(r['image_path']).name}")
            print()
    
    # Analyze similarity distributions
    all_similarities = [s for r in results for s in r['similarities']]
    print(f"Similarity Statistics:")
    print(f"  Min: {min(all_similarities):.4f}")
    print(f"  Max: {max(all_similarities):.4f}")
    print(f"  Mean: {np.mean(all_similarities):.4f}")
    print(f"  Std: {np.std(all_similarities):.4f}")
    
    # Save detailed results
    with open("clip_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: clip_analysis_results.json")
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'error_types': dict(error_types),
        'results': results
    }

if __name__ == "__main__":
    analyze_model_performance()