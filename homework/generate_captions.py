from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # Import required functions from generate_qa
    from .generate_qa import extract_kart_objects, extract_track_info
    
    captions = []
    
    # Extract information from the info file
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track = extract_track_info(info_path)
    
    if not karts:
        return captions
    
    # Find ego kart
    ego = next(k for k in karts if k["is_center_kart"])
    ego_name = ego["kart_name"]
    ego_x, ego_y = ego["center"]
    
    # 1. Ego car caption
    captions.append(f"{ego_name} is the ego car.")
    
    # 2. Counting caption
    captions.append(f"There are {len(karts)} karts in the scene.")
    
    # 3. Track name caption
    captions.append(f"The track is {track}.")
    
    # 4. Enhanced relative position captions for other karts (following HW4 tips)
    for kart in karts:
        if kart["is_center_kart"]:
            continue
            
        kart_name = kart["kart_name"]
        kart_x, kart_y = kart["center"]
        
        # More detailed relative positioning (front/back takes priority over left/right)
        if abs(kart_y - ego_y) > abs(kart_x - ego_x):
            # Vertical positioning is more significant
            if kart_y < ego_y:
                if abs(kart_x - ego_x) > 10:  # Add horizontal detail if significant
                    horizontal = "to the left" if kart_x < ego_x else "to the right"
                    position = f"in front and {horizontal}"
                else:
                    position = "in front"
            else:
                if abs(kart_x - ego_x) > 10:
                    horizontal = "to the left" if kart_x < ego_x else "to the right"
                    position = f"behind and {horizontal}"
                else:
                    position = "behind"
        else:
            # Horizontal positioning is more significant
            if kart_x < ego_x:
                position = "to the left"
            else:
                position = "to the right"
            
        # Generate multiple caption variants for better training
        captions.append(f"{kart_name} is {position} of {ego_name}.")
        captions.append(f"{kart_name} is positioned {position} of the ego car.")
        
        # Add distance-based descriptions
        distance = ((kart_x - ego_x)**2 + (kart_y - ego_y)**2)**0.5
        if distance < 30:
            captions.append(f"{kart_name} is close to {ego_name}.")
        elif distance > 80:
            captions.append(f"{kart_name} is far from {ego_name}.")
    
    # 5. Scene composition captions
    if len(karts) > 1:
        other_karts = [k["kart_name"] for k in karts if not k["is_center_kart"]]
        if len(other_karts) == 1:
            captions.append(f"{ego_name} is racing against {other_karts[0]}.")
        else:
            captions.append(f"{ego_name} is racing against {', '.join(other_karts[:-1])} and {other_karts[-1]}.")
            
    # 6. Explicit naming in context
    captions.append(f"The ego car {ego_name} is on the {track} track.")
    if len(karts) > 2:
        captions.append(f"Multiple karts including {ego_name} are visible in this racing scene.")
    
    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def generate_dataset():
    """Generate captions for all training images and save to JSON file"""
    import json
    from pathlib import Path
    
    data_dir = Path("data/train")
    
    # Find all info.json files
    info_files = list(data_dir.glob("*_info.json"))
    print(f"Found {len(info_files)} info files")
    
    all_captions = []
    
    for info_file in info_files:
        print(f"Processing {info_file.name}...")
        
        # Extract frame ID from info file name
        frame_id = info_file.stem.replace("_info", "")
        
        # Find all corresponding image files for this frame
        image_pattern = f"{frame_id}_*_im.jpg"
        image_files = list(data_dir.glob(image_pattern))
        
        for image_file in image_files:
            # Extract view index from image filename
            parts = image_file.stem.split("_")
            if len(parts) >= 2:
                try:
                    view_index = int(parts[1])
                    
                    # Generate captions for this view
                    captions = generate_caption(str(info_file), view_index)
                    
                    # Add each caption as a separate entry
                    for caption in captions:
                        caption_entry = {
                            "image_file": f"train/{image_file.name}",
                            "caption": caption
                        }
                        all_captions.append(caption_entry)
                        
                except ValueError:
                    print(f"Could not parse view index from {image_file.name}")
                    continue
                except Exception as e:
                    print(f"Error processing {image_file.name}: {e}")
                    continue
    
    print(f"Generated {len(all_captions)} caption entries")
    
    # Save to JSON file
    output_file = data_dir / "train_captions.json"
    with open(output_file, 'w') as f:
        json.dump(all_captions, f, indent=2)
    
    print(f"Saved captions to {output_file}")
    
    # Show some examples
    print("\nExample captions:")
    for i, caption in enumerate(all_captions[:10]):
        print(f"{i+1}. {caption['image_file']}: {caption['caption']}")


def main():
    fire.Fire({"check": check_caption, "generate": generate_dataset})


if __name__ == "__main__":
    main()
