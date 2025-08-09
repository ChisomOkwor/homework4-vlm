#!/usr/bin/env python3
"""
Monitor both CLIP training processes and compare results.
"""

import time
import json
from pathlib import Path
from homework.clip import evaluate_checkpoint as eval_original
from homework.clip_enhanced import evaluate_checkpoint as eval_enhanced

def monitor_training():
    """Monitor both training processes."""
    base_dir = Path(__file__).parent / "homework"
    
    conservative_dir = base_dir / "clip"
    enhanced_dir = base_dir / "clip_enhanced"
    
    print("🔍 MONITORING DUAL CLIP TRAINING")
    print("=" * 50)
    print("Conservative: Continue from 37% → Better")
    print("Enhanced: Fresh start with improved architecture")
    print("=" * 50)
    
    last_conservative_checkpoint = None
    last_enhanced_checkpoint = None
    
    while True:
        try:
            print(f"\n⏰ {time.strftime('%H:%M:%S')} - Checking progress...")
            
            # Check conservative training
            conservative_checkpoints = []
            if conservative_dir.exists():
                conservative_checkpoints = [
                    d for d in conservative_dir.iterdir() 
                    if d.is_dir() and d.name.startswith("checkpoint-")
                ]
                
            # Check enhanced training  
            enhanced_checkpoints = []
            if enhanced_dir.exists():
                enhanced_checkpoints = [
                    d for d in enhanced_dir.iterdir()
                    if d.is_dir() and d.name.startswith("checkpoint-")
                ]
            
            print(f"Conservative checkpoints: {len(conservative_checkpoints)}")
            print(f"Enhanced checkpoints: {len(enhanced_checkpoints)}")
            
            # Evaluate latest conservative checkpoint
            if conservative_checkpoints:
                latest_conservative = max(conservative_checkpoints, key=lambda x: int(x.name.split("-")[1]))
                if latest_conservative != last_conservative_checkpoint:
                    print(f"\n🔄 Evaluating conservative: {latest_conservative.name}")
                    try:
                        acc = eval_original(str(latest_conservative))
                        print(f"   Conservative accuracy: {acc:.4f}")
                        
                        # Check if best checkpoint exists
                        best_conservative = conservative_dir / "best_checkpoint"
                        if best_conservative.exists():
                            best_acc = eval_original(str(best_conservative))
                            print(f"   Conservative BEST: {best_acc:.4f}")
                    except Exception as e:
                        print(f"   Error evaluating conservative: {e}")
                    last_conservative_checkpoint = latest_conservative
            
            # Evaluate latest enhanced checkpoint
            if enhanced_checkpoints:
                latest_enhanced = max(enhanced_checkpoints, key=lambda x: int(x.name.split("-")[1]))
                if latest_enhanced != last_enhanced_checkpoint:
                    print(f"\n🔄 Evaluating enhanced: {latest_enhanced.name}")
                    try:
                        acc = eval_enhanced(str(latest_enhanced))
                        print(f"   Enhanced accuracy: {acc:.4f}")
                        
                        # Check if best checkpoint exists
                        best_enhanced = enhanced_dir / "best_checkpoint"
                        if best_enhanced.exists():
                            best_acc = eval_enhanced(str(best_enhanced))
                            print(f"   Enhanced BEST: {best_acc:.4f}")
                    except Exception as e:
                        print(f"   Error evaluating enhanced: {e}")
                    last_enhanced_checkpoint = latest_enhanced
            
            # Compare best results
            conservative_best = conservative_dir / "best_checkpoint"
            enhanced_best = enhanced_dir / "best_checkpoint"
            
            if conservative_best.exists() and enhanced_best.exists():
                try:
                    conservative_best_acc = eval_original(str(conservative_best))
                    enhanced_best_acc = eval_enhanced(str(enhanced_best))
                    
                    print(f"\n📊 COMPARISON:")
                    print(f"   Conservative (continue 37%): {conservative_best_acc:.4f}")
                    print(f"   Enhanced (fresh start): {enhanced_best_acc:.4f}")
                    
                    if enhanced_best_acc > conservative_best_acc:
                        print(f"   🏆 Enhanced is winning by +{enhanced_best_acc - conservative_best_acc:.4f}")
                    elif conservative_best_acc > enhanced_best_acc:
                        print(f"   🏆 Conservative is winning by +{conservative_best_acc - enhanced_best_acc:.4f}")
                    else:
                        print(f"   🤝 They're tied!")
                        
                except Exception as e:
                    print(f"   Error comparing: {e}")
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print(f"\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"Error in monitoring: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_training()