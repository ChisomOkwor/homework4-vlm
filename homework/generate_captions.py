#!/usr/bin/env python3
"""
Generate captions for CLIP training from SuperTuxKart data.
This script creates image-caption pairs for contrastive learning.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Define object types for captions
OBJECT_TYPES = {
    1: "kart",
    2: "track boundary", 
    3: "track element",
    4: "special element 1",
    5: "special element 2", 
    6: "special element 3"
}

# Define kart names for more descriptive captions
KART_NAMES = [
    "emule", "sara", "peach", "yoshi", "mario", "luigi", "bowser", "donkey kong",
    "wario", "waluigi", "toad", "daisy", "rosalina", "link", "zelda", "ganon"
]

def extract_kart_objects(info: Dict[str, Any], view_index: int) -> List[Dict[str, Any]]:
    """Extract kart objects from detection data."""
    karts = []
    
    if "detections" not in info or view_index >= len(info["detections"]):
        return karts
    
    frame_detections = info["detections"][view_index]
    
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        
        # Only process karts (class_id == 1)
        if int(class_id) == 1:
            karts.append({
                "class_id": int(class_id),
                "track_id": int(track_id),
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2)
            })
    
    return karts

def generate_captions(info: Dict[str, Any], view_index: int, base_name: str) -> List[Dict[str, Any]]:
    """Generate captions for a single image."""
    captions = []
    
    # Extract kart information
    karts = extract_kart_objects(info, view_index)
    
    # Get kart names if available
    kart_names = info.get("karts", [])
    
    # Generate various types of captions
    
    # 1. Count-based captions
    if len(karts) == 0:
        captions.append("There are no karts in the scene.")
    elif len(karts) == 1:
        captions.append("There is 1 kart in the scene.")
    else:
        captions.append(f"There are {len(karts)} karts in the scene.")
    
    # 2. Position-based captions
    if len(karts) > 0:
        # Find the kart closest to center (ego kart)
        center_x, center_y = 300, 200  # Image center
        ego_kart = min(karts, key=lambda k: abs((k['x1'] + k['x2'])/2 - center_x) + abs((k['y1'] + k['y2'])/2 - center_y))
        
        # Ego kart description
        if ego_kart['track_id'] == 0:
            captions.append("The ego kart is in the center of the scene.")
        else:
            captions.append("A kart is in the center of the scene.")
    
    # 3. Track-based captions
    if "track" in info:
        track_name = info["track"]
        captions.append(f"The track is {track_name}.")
    
    # 4. Kart name captions (if available)
    if kart_names and len(kart_names) > view_index:
        kart_name = kart_names[view_index]
        if kart_name:
            captions.append(f"{kart_name} is the ego car.")
    
    # 5. Scene description captions
    if len(karts) > 1:
        captions.append("Multiple karts are racing on the track.")
    elif len(karts) == 1:
        captions.append("A single kart is on the track.")
    
    # 6. Spatial relationship captions
    if len(karts) >= 2:
        # Find karts on left and right
        left_karts = [k for k in karts if (k['x1'] + k['x2'])/2 < 250]
        right_karts = [k for k in karts if (k['x1'] + k['x2'])/2 > 350]
        
        if left_karts:
            captions.append("There are karts on the left side.")
        if right_karts:
            captions.append("There are karts on the right side.")
    
    # 7. Distance-based captions
    if len(karts) >= 2:
        # Find closest kart to ego
        ego_kart = next((k for k in karts if k['track_id'] == 0), karts[0])
        other_karts = [k for k in karts if k['track_id'] != 0]
        
        if other_karts:
            closest_kart = min(other_karts, key=lambda k: 
                abs((k['x1'] + k['x2'])/2 - (ego_kart['x1'] + ego_kart['x2'])/2) + 
                abs((k['y1'] + k['y2'])/2 - (ego_kart['y1'] + ego_kart['y2'])/2))
            
            distance = abs((closest_kart['x1'] + closest_kart['x2'])/2 - (ego_kart['x1'] + ego_kart['x2'])/2)
            if distance < 100:
                captions.append("A kart is very close to the ego kart.")
            elif distance < 200:
                captions.append("A kart is nearby the ego kart.")
    
    return captions

def generate_caption_dataset(data_dir: Path, output_file: str = "captions.json"):
    """Generate caption dataset from SuperTuxKart data."""
    captions = []
    
    # Process all info files in the data directory
    info_files = list(data_dir.glob("*_info.json"))
    
    # Determine the directory name for image paths
    dir_name = data_dir.name
    
    print(f"Found {len(info_files)} info files to process...")
    
    for info_file in info_files:
        try:
            with open(info_file, 'r') as f:
                info = json.load(f)
            
            # Extract base name (remove _info.json)
            base_name = info_file.stem.replace('_info', '')
            
            # Process each view
            num_views = len(info.get("detections", []))
            
            for view_index in range(num_views):
                # Generate captions for this view
                view_captions = generate_captions(info, view_index, base_name)
                
                # Create image file path - use the correct directory structure
                image_file = f"{dir_name}/{base_name}_{view_index:02d}_im.jpg"
                
                # Add each caption as a separate entry
                for caption in view_captions:
                    captions.append({
                        "image_file": image_file,
                        "caption": caption
                    })
        
        except Exception as e:
            print(f"Error processing {info_file}: {e}")
            continue
    
    # Save captions to file
    output_path = data_dir / output_file
    with open(output_path, 'w') as f:
        json.dump(captions, f, indent=2)
    
    print(f"Generated {len(captions)} captions saved to {output_path}")
    return captions

def main():
    """Main function to generate caption dataset."""
    import fire
    
    def generate_captions_main(
        data_dir: str = "data/valid",
        output_file: str = "captions.json"
    ):
        """Generate captions for CLIP training."""
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"Data directory {data_path} does not exist!")
            return
        
        captions = generate_caption_dataset(data_path, output_file)
        print(f"Successfully generated {len(captions)} captions!")
    
    fire.Fire(generate_captions_main)

if __name__ == "__main__":
    main()
