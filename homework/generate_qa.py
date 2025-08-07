import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os
from tqdm import tqdm

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    with open(info_path) as f:
        info = json.load(f)

    karts = []
    frame_detections = info["detections"][view_index]
    kart_names = info["karts"]  # Use "karts" field instead of "names"

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        if int(class_id) != 1:  # Only keep kart objects
            continue

        # Rescale coords
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y

        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2

        # Map track_id to kart name using index
        track_id_int = int(track_id)
        kart_name = kart_names[track_id_int] if track_id_int < len(kart_names) else f"kart_{track_id}"

        karts.append({
            "instance_id": track_id_int,
            "kart_name": kart_name,
            "center": (center_x, center_y),
        })

    # Identify center (ego) kart as closest to image center
    image_center = (img_width / 2, img_height / 2)
    for kart in karts:
        dx = kart["center"][0] - image_center[0]
        dy = kart["center"][1] - image_center[1]
        kart["distance_to_center"] = dx**2 + dy**2

    if karts:
        ego_id = min(karts, key=lambda x: x["distance_to_center"])["instance_id"]
        for kart in karts:
            kart["is_center_kart"] = (kart["instance_id"] == ego_id)

    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path) as f:
        info = json.load(f)
    return info.get("track", "unknown")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question, answer, and image_file
    """
    # Extract kart objects
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    
    # Load track information
    with open(info_path, 'r') as f:
        info = json.load(f)
    track_name = info["track"]
    
    # Get image file name
    base_name = os.path.splitext(os.path.basename(info_path))[0]
    # Remove "_info" suffix and format correctly
    base_name = base_name.replace("_info", "")
    image_file = f"valid/{base_name}_{view_index:02d}_im.jpg"
    
    qa_pairs = []
    
    # 1. Ego kart identification (MULTIPLE VARIATIONS)
    if karts:
        ego_kart = None
        for kart in karts:
            if kart["is_center_kart"]:
                ego_kart = kart
                break
        
        if ego_kart:
            # Multiple variations of ego kart questions
            ego_questions = [
                f"What kart is the ego car?",
                f"Which kart is the player driving?",
                f"What is the name of the ego kart?",
                f"Identify the ego car in this scene.",
                f"What kart am I controlling?",
                f"Which kart represents the player?",
                f"What is the ego kart called?",
                f"Name the kart that is the ego car.",
                f"What kart is the player using?",
                f"Which kart is the main character?",
                f"Identify the player's kart.",
                f"What kart am I driving?",
                f"Name the ego kart in this scene.",
                f"Which kart is the ego car?",
                f"What is the player's kart called?"
            ]
            
            for question in ego_questions:
                qa_pairs.append({
                    "question": question,
                    "answer": ego_kart["kart_name"],
                    "image_file": image_file
                })
    
    # 2. Total karts counting (MULTIPLE VARIATIONS)
    total_karts = len(karts)
    counting_questions = [
        f"How many karts are there in the scenario?",
        f"What is the total number of karts in this scene?",
        f"Count the number of karts visible.",
        f"How many karts can you see?",
        f"What is the kart count in this image?",
        f"Number of karts in the scene?",
        f"Total karts present?",
        f"How many racing karts are there?"
    ]
    
    for question in counting_questions:
        qa_pairs.append({
            "question": question,
            "answer": str(total_karts),
            "image_file": image_file
        })
    
    # 3. Track recognition (MULTIPLE VARIATIONS - CRITICAL FOR ACCURACY)
    track_questions = [
        f"What track is this?",
        f"Which SuperTuxKart track is shown?",
        f"Name the track in this image.",
        f"What is the name of this racing track?",
        f"Identify the SuperTuxKart track.",
        f"Which track are we racing on?",
        f"What track is being displayed?",
        f"Name this racing circuit.",
        f"What is the track called?",
        f"Identify the racing track.",
        f"What SuperTuxKart track is this?",
        f"Name the racing track shown.",
        f"Which track is displayed?",
        f"What track are we on?",
        f"Identify this SuperTuxKart track."
    ]
    
    for question in track_questions:
        qa_pairs.append({
            "question": question,
            "answer": track_name,
            "image_file": image_file
        })
    
    # 4. Relative positioning (MULTIPLE VARIATIONS - CRITICAL FOR SPATIAL REASONING)
    if len(karts) > 1:
        ego_kart = None
        other_karts = []
        
        for kart in karts:
            if kart["is_center_kart"]:
                ego_kart = kart
            else:
                other_karts.append(kart)
        
        if ego_kart and other_karts:
            for other_kart in other_karts:
                # Determine relative position
                ego_x, ego_y = ego_kart["center"]
                other_x, other_y = other_kart["center"]
                
                # Front/behind (based on Y coordinate - lower Y is "front")
                if other_y < ego_y - 10:
                    position = "front"
                elif other_y > ego_y + 10:
                    position = "behind"
                else:
                    position = "beside"
                
                # Left/right (based on X coordinate)
                if other_x < ego_x - 10:
                    side = "left"
                elif other_x > ego_x + 10:
                    side = "right"
                else:
                    side = "center"
                
                # Generate comprehensive spatial reasoning questions
                spatial_questions = [
                    f"Where is {other_kart['kart_name']} relative to the ego car?",
                    f"What is the position of {other_kart['kart_name']} compared to the ego car?",
                    f"How is {other_kart['kart_name']} positioned relative to the ego car?",
                    f"Where is {other_kart['kart_name']} located in relation to the ego car?",
                    f"What is {other_kart['kart_name']}'s position relative to the ego car?",
                    f"Where can {other_kart['kart_name']} be found relative to the ego car?",
                    f"What is the relative position of {other_kart['kart_name']} to the ego car?",
                    f"Where is {other_kart['kart_name']} situated relative to the ego car?",
                    f"What is {other_kart['kart_name']}'s location relative to the ego car?",
                    f"Where is {other_kart['kart_name']} positioned relative to the ego car?",
                    f"Is {other_kart['kart_name']} to the left or right of the ego car?",
                    f"Is {other_kart['kart_name']} in front of or behind the ego car?",
                    f"Is {other_kart['kart_name']} left or right of the ego car?",
                    f"Is {other_kart['kart_name']} front or behind the ego car?",
                    f"Which side is {other_kart['kart_name']} on relative to the ego car?",
                    f"Is {other_kart['kart_name']} ahead or behind the ego car?",
                    f"Is {other_kart['kart_name']} to the left or right side of the ego car?",
                    f"Is {other_kart['kart_name']} in front or behind the ego car?",
                    f"Which direction is {other_kart['kart_name']} from the ego car?",
                    f"Where is {other_kart['kart_name']} relative to the player's kart?"
                ]
                
                # Determine the correct answer based on position and side
                if position == "front" and side == "left":
                    correct_answer = "front and left"
                elif position == "front" and side == "right":
                    correct_answer = "front and right"
                elif position == "behind" and side == "left":
                    correct_answer = "back and left"
                elif position == "behind" and side == "right":
                    correct_answer = "back and right"
                elif position == "front":
                    correct_answer = "front"
                elif position == "behind":
                    correct_answer = "back"
                elif side == "left":
                    correct_answer = "left"
                elif side == "right":
                    correct_answer = "right"
                else:
                    correct_answer = "beside"
                
                # Add spatial reasoning questions
                for question in spatial_questions:
                    qa_pairs.append({
                        "question": question,
                        "answer": correct_answer,
                        "image_file": image_file
                    })
    
    # 5. Spatial counting questions (MULTIPLE VARIATIONS - CRITICAL FOR SPATIAL COUNTING)
    if len(karts) > 1:
        ego_kart = None
        other_karts = []
        
        for kart in karts:
            if kart["is_center_kart"]:
                ego_kart = kart
            else:
                other_karts.append(kart)
        
        if ego_kart and other_karts:
            # Count karts in front
            karts_in_front = sum(1 for kart in other_karts if kart["center"][1] < ego_kart["center"][1] - 10)
            front_count_questions = [
                "How many karts are in front of the ego car?",
                "How many karts are ahead of the ego car?",
                "How many karts are in front of the player?",
                "How many karts are ahead of the player?",
                "How many karts are in front of the ego kart?",
                "How many karts are ahead of the ego kart?",
                "How many karts are in front of the player's kart?",
                "How many karts are ahead of the player's kart?",
                "How many karts are in front of the main kart?",
                "How many karts are ahead of the main kart?"
            ]
            for question in front_count_questions:
                qa_pairs.append({
                    "question": question,
                    "answer": str(karts_in_front),
                    "image_file": image_file
                })
            
            # Count karts behind
            karts_behind = sum(1 for kart in other_karts if kart["center"][1] > ego_kart["center"][1] + 10)
            behind_count_questions = [
                "How many karts are behind the ego car?",
                "How many karts are behind the player?",
                "How many karts are behind the ego kart?",
                "How many karts are behind the player's kart?",
                "How many karts are behind the main kart?",
                "How many karts are at the back of the ego car?",
                "How many karts are at the back of the player?",
                "How many karts are at the back of the ego kart?",
                "How many karts are at the back of the player's kart?",
                "How many karts are at the back of the main kart?"
            ]
            for question in behind_count_questions:
                qa_pairs.append({
                    "question": question,
                    "answer": str(karts_behind),
                    "image_file": image_file
                })
            
            # Count karts to the left
            karts_left = sum(1 for kart in other_karts if kart["center"][0] < ego_kart["center"][0] - 10)
            left_count_questions = [
                "How many karts are to the left of the ego car?",
                "How many karts are to the left of the player?",
                "How many karts are to the left of the ego kart?",
                "How many karts are to the left of the player's kart?",
                "How many karts are to the left of the main kart?",
                "How many karts are on the left side of the ego car?",
                "How many karts are on the left side of the player?",
                "How many karts are on the left side of the ego kart?",
                "How many karts are on the left side of the player's kart?",
                "How many karts are on the left side of the main kart?"
            ]
            for question in left_count_questions:
                qa_pairs.append({
                    "question": question,
                    "answer": str(karts_left),
                    "image_file": image_file
                })
            
            # Count karts to the right
            karts_right = sum(1 for kart in other_karts if kart["center"][0] > ego_kart["center"][0] + 10)
            right_count_questions = [
                "How many karts are to the right of the ego car?",
                "How many karts are to the right of the player?",
                "How many karts are to the right of the ego kart?",
                "How many karts are to the right of the player's kart?",
                "How many karts are to the right of the main kart?",
                "How many karts are on the right side of the ego car?",
                "How many karts are on the right side of the player?",
                "How many karts are on the right side of the ego kart?",
                "How many karts are on the right side of the player's kart?",
                "How many karts are on the right side of the main kart?"
            ]
            for question in right_count_questions:
                qa_pairs.append({
                    "question": question,
                    "answer": str(karts_right),
                    "image_file": image_file
                })
    
    return qa_pairs


def generate_complete_dataset():
    """
    Generate QA pairs for all validation files and save to train directory.
    """
    valid_dir = Path("data/valid")
    all_qa_pairs = []
    
    # Process all JSON files in valid directory
    for json_file in valid_dir.glob("*_info.json"):
        print(f"Processing {json_file.name}...")
        
        # Process all 10 views for each file
        for view_index in range(10):
            try:
                qa_pairs = generate_qa_pairs(str(json_file), view_index)
                all_qa_pairs.extend(qa_pairs)
            except Exception as e:
                print(f"Error processing {json_file.name} view {view_index}: {e}")
                continue
    
    # Save to train directory
    output_file = Path("data/train/balanced_qa_pairs.json")
    with open(output_file, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"Generated {len(all_qa_pairs)} QA pairs and saved to {output_file}")
    return all_qa_pairs

def generate_focused_dataset(data_dir: Path = None, max_qa_pairs: int = 100000):
    """
    Generate a smaller, focused dataset with only the most important spatial reasoning questions.
    This is optimized for faster training on Mac.
    """
    if data_dir is None:
        data_dir = Path("data")
    
    print(f"Generating focused dataset with max {max_qa_pairs} QA pairs...")
    
    # Get all info files
    valid_dir = data_dir / "valid"
    info_files = list(valid_dir.glob("*_info.json"))
    
    qa_pairs = []
    qa_count = 0
    
    for info_file in tqdm(info_files, desc="Processing files"):
        if qa_count >= max_qa_pairs:
            break
            
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        base_name = info_file.stem.replace('_info', '')
        
        # Process each view
        for view_index in range(len(info["karts"])):
            if qa_count >= max_qa_pairs:
                break
                
            # Extract kart objects directly from info
            kart_names = info["karts"]
            kart_objects = []
            
            # Find ego kart (first kart in the list)
            ego_kart_name = kart_names[0]
            ego_kart = {
                'kart_name': ego_kart_name,
                'center_x': 400,  # Approximate center
                'center_y': 300
            }
            kart_objects.append(ego_kart)
            
            # Add other karts with approximate positions
            for i, kart_name in enumerate(kart_names[1:], 1):
                # Simple positioning based on index
                kart_objects.append({
                    'kart_name': kart_name,
                    'center_x': 400 + (i * 100),  # Spread horizontally
                    'center_y': 300 + (i * 50)    # Spread vertically
                })
            
            if not kart_objects:
                continue
            
            ego_kart = kart_objects[0]  # First kart is ego
            other_karts = kart_objects[1:]
            
            image_file = f"valid/{base_name}_{view_index:02d}_im.jpg"
            
            # Generate focused QA pairs - only the most important ones
            # 1. Ego kart identification (critical)
            qa_pairs.append({
                "question": f"What kart is the ego car?",
                "answer": ego_kart['kart_name'],
                "image_file": image_file
            })
            qa_count += 1
            
            # 2. Total kart counting (critical)
            qa_pairs.append({
                "question": f"How many karts are there in the scenario?",
                "answer": str(len(kart_objects)),
                "image_file": image_file
            })
            qa_count += 1
            
            # 3. Track recognition (critical)
            qa_pairs.append({
                "question": f"What track is this?",
                "answer": info["track"],
                "image_file": image_file
            })
            qa_count += 1
            
            # 4. Focus on spatial reasoning - only the most important variations
            for other_kart in other_karts:
                if qa_count >= max_qa_pairs:
                    break
                    
                # Determine position relative to ego
                dx = other_kart['center_x'] - ego_kart['center_x']
                dy = other_kart['center_y'] - ego_kart['center_y']
                
                # Simple spatial questions
                if abs(dx) > 50:  # Significant horizontal difference
                    side = "left" if dx < 0 else "right"
                    qa_pairs.append({
                        "question": f"Is {other_kart['kart_name']} to the left or right of the ego car?",
                        "answer": side,
                        "image_file": image_file
                    })
                    qa_count += 1
                
                if abs(dy) > 50:  # Significant vertical difference
                    position = "front" if dy < 0 else "behind"
                    qa_pairs.append({
                        "question": f"Is {other_kart['kart_name']} in front of or behind the ego car?",
                        "answer": position,
                        "image_file": image_file
                    })
                    qa_count += 1
                
                # Spatial counting - only one variation
                if dx < -50:  # Left
                    qa_pairs.append({
                        "question": f"How many karts are to the left of the ego car?",
                        "answer": str(len([k for k in other_karts if k['center_x'] < ego_kart['center_x'] - 50])),
                        "image_file": image_file
                    })
                    qa_count += 1
                
                if dx > 50:  # Right
                    qa_pairs.append({
                        "question": f"How many karts are to the right of the ego car?",
                        "answer": str(len([k for k in other_karts if k['center_x'] > ego_kart['center_x'] + 50])),
                        "image_file": image_file
                    })
                    qa_count += 1
    
    # Save focused dataset
    output_file = data_dir / "train" / "focused_qa_pairs.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    
    print(f"Generated {len(qa_pairs)} focused QA pairs and saved to {output_file}")
    return output_file

def generate_spatial_focused_dataset(data_dir: Path = None, max_qa_pairs: int = 150000):
    """
    Generate a dataset specifically focused on spatial reasoning problems.
    This targets the specific weaknesses: defaulting to "behind", answering "0" for counting.
    """
    if data_dir is None:
        data_dir = Path("data")
    
    print(f"Generating spatial reasoning focused dataset with max {max_qa_pairs} QA pairs...")
    
    # Get all info files
    valid_dir = data_dir / "valid"
    info_files = list(valid_dir.glob("*_info.json"))
    
    qa_pairs = []
    qa_count = 0
    
    for info_file in tqdm(info_files, desc="Processing files"):
        if qa_count >= max_qa_pairs:
            break
            
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        base_name = info_file.stem.replace('_info', '')
        
        # Process each view
        for view_index in range(len(info["karts"])):
            if qa_count >= max_qa_pairs:
                break
                
            # Extract kart objects with real bounding box data
            kart_objects = extract_kart_objects_from_file(info_file, view_index)
            if not kart_objects:
                continue
            
            ego_kart = kart_objects[0]  # First kart is ego
            other_karts = kart_objects[1:]
            
            image_file = f"valid/{base_name}_{view_index:02d}_im.jpg"
            
            # 1. Basic questions (keep some for balance)
            qa_pairs.append({
                "question": f"What kart is the ego car?",
                "answer": ego_kart['kart_name'],
                "image_file": image_file
            })
            qa_count += 1
            
            qa_pairs.append({
                "question": f"How many karts are there in the scenario?",
                "answer": str(len(kart_objects)),
                "image_file": image_file
            })
            qa_count += 1
            
            qa_pairs.append({
                "question": f"What track is this?",
                "answer": info["track"],
                "image_file": image_file
            })
            qa_count += 1
            
            # 2. SPATIAL REASONING FOCUS - Multiple variations for each kart
            for other_kart in other_karts:
                if qa_count >= max_qa_pairs:
                    break
                    
                # Determine real position relative to ego
                dx = other_kart['center_x'] - ego_kart['center_x']
                dy = other_kart['center_y'] - ego_kart['center_y']
                
                # Multiple spatial reasoning questions for each kart
                spatial_questions = []
                
                # Horizontal positioning (left/right)
                if abs(dx) > 30:  # Significant horizontal difference
                    side = "left" if dx < 0 else "right"
                    spatial_questions.extend([
                        f"Is {other_kart['kart_name']} to the left or right of the ego car?",
                        f"Which side is {other_kart['kart_name']} on relative to the ego car?",
                        f"Is {other_kart['kart_name']} left or right of the ego car?",
                        f"Is {other_kart['kart_name']} to the left or right side of the ego car?",
                        f"Where is {other_kart['kart_name']} positioned relative to the ego car?",
                    ])
                
                # Vertical positioning (front/behind)
                if abs(dy) > 30:  # Significant vertical difference
                    position = "front" if dy < 0 else "behind"
                    spatial_questions.extend([
                        f"Is {other_kart['kart_name']} in front of or behind the ego car?",
                        f"Is {other_kart['kart_name']} front or behind the ego car?",
                        f"Is {other_kart['kart_name']} ahead or behind the ego car?",
                        f"Where is {other_kart['kart_name']} located in relation to the ego car?",
                        f"What is {other_kart['kart_name']}'s position relative to the ego car?",
                    ])
                
                # Combined positioning (front/back and left/right)
                if abs(dx) > 30 and abs(dy) > 30:
                    if dx < 0 and dy < 0:
                        combined = "front and left"
                    elif dx > 0 and dy < 0:
                        combined = "front and right"
                    elif dx < 0 and dy > 0:
                        combined = "back and left"
                    else:
                        combined = "back and right"
                    
                    spatial_questions.extend([
                        f"Where is {other_kart['kart_name']} relative to the ego car?",
                        f"What is the position of {other_kart['kart_name']} compared to the ego car?",
                        f"How is {other_kart['kart_name']} positioned relative to the ego car?",
                        f"Where is {other_kart['kart_name']} located in relation to the ego car?",
                        f"What is {other_kart['kart_name']}'s location relative to the ego car?",
                    ])
                
                # Add all spatial questions
                for question in spatial_questions:
                    if qa_count >= max_qa_pairs:
                        break
                    
                    # Determine correct answer based on question type
                    if "left or right" in question.lower():
                        correct_answer = "left" if dx < 0 else "right"
                    elif "front or behind" in question.lower() or "ahead or behind" in question.lower():
                        correct_answer = "front" if dy < 0 else "behind"
                    elif "where is" in question.lower() or "position" in question.lower() or "location" in question.lower():
                        if abs(dx) > 30 and abs(dy) > 30:
                            if dx < 0 and dy < 0:
                                correct_answer = "front and left"
                            elif dx > 0 and dy < 0:
                                correct_answer = "front and right"
                            elif dx < 0 and dy > 0:
                                correct_answer = "back and left"
                            else:
                                correct_answer = "back and right"
                        elif abs(dx) > 30:
                            correct_answer = "left" if dx < 0 else "right"
                        else:
                            correct_answer = "front" if dy < 0 else "behind"
                    else:
                        continue
                    
                    qa_pairs.append({
                        "question": question,
                        "answer": correct_answer,
                        "image_file": image_file
                    })
                    qa_count += 1
            
            # 3. SPATIAL COUNTING FOCUS - Multiple variations
            if len(other_karts) > 0:
                # Count karts to the left
                left_count = len([k for k in other_karts if k['center_x'] < ego_kart['center_x'] - 30])
                if left_count > 0:  # Only ask if there are actually karts to the left
                    counting_questions = [
                        f"How many karts are to the left of the ego car?",
                        f"How many karts are on the left side of the ego car?",
                        f"How many karts are positioned to the left of the ego car?",
                        f"Count the karts to the left of the ego car.",
                        f"How many karts can be found to the left of the ego car?",
                    ]
                    for question in counting_questions:
                        if qa_count >= max_qa_pairs:
                            break
                        qa_pairs.append({
                            "question": question,
                            "answer": str(left_count),
                            "image_file": image_file
                        })
                        qa_count += 1
                
                # Count karts to the right
                right_count = len([k for k in other_karts if k['center_x'] > ego_kart['center_x'] + 30])
                if right_count > 0:  # Only ask if there are actually karts to the right
                    counting_questions = [
                        f"How many karts are to the right of the ego car?",
                        f"How many karts are on the right side of the ego car?",
                        f"How many karts are positioned to the right of the ego car?",
                        f"Count the karts to the right of the ego car.",
                        f"How many karts can be found to the right of the ego car?",
                    ]
                    for question in counting_questions:
                        if qa_count >= max_qa_pairs:
                            break
                        qa_pairs.append({
                            "question": question,
                            "answer": str(right_count),
                            "image_file": image_file
                        })
                        qa_count += 1
                
                # Count karts in front
                front_count = len([k for k in other_karts if k['center_y'] < ego_kart['center_y'] - 30])
                if front_count > 0:  # Only ask if there are actually karts in front
                    counting_questions = [
                        f"How many karts are in front of the ego car?",
                        f"How many karts are ahead of the ego car?",
                        f"How many karts are positioned in front of the ego car?",
                        f"Count the karts in front of the ego car.",
                        f"How many karts can be found in front of the ego car?",
                    ]
                    for question in counting_questions:
                        if qa_count >= max_qa_pairs:
                            break
                        qa_pairs.append({
                            "question": question,
                            "answer": str(front_count),
                            "image_file": image_file
                        })
                        qa_count += 1
                
                # Count karts behind
                behind_count = len([k for k in other_karts if k['center_y'] > ego_kart['center_y'] + 30])
                if behind_count > 0:  # Only ask if there are actually karts behind
                    counting_questions = [
                        f"How many karts are behind the ego car?",
                        f"How many karts are positioned behind the ego car?",
                        f"Count the karts behind the ego car.",
                        f"How many karts can be found behind the ego car?",
                        f"How many karts are at the back of the ego car?",
                    ]
                    for question in counting_questions:
                        if qa_count >= max_qa_pairs:
                            break
                        qa_pairs.append({
                            "question": question,
                            "answer": str(behind_count),
                            "image_file": image_file
                        })
                        qa_count += 1
    
    # Save spatial focused dataset
    output_file = data_dir / "train" / "spatial_focused_qa_pairs.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    
    print(f"Generated {len(qa_pairs)} spatial reasoning focused QA pairs and saved to {output_file}")
    return output_file

def extract_kart_objects_from_file(info_file: Path, view_index: int):
    """
    Extract kart objects from a specific info file and view index.
    This uses the real bounding box data for accurate spatial reasoning.
    """
    try:
        with open(info_file) as f:
            info = json.load(f)
        
        kart_names = info["karts"]
        kart_objects = []
        
        # Find ego kart (first kart in the list)
        ego_kart_name = kart_names[0]
        
        # Get bounding box data for ego kart
        ego_bbox = None
        for detection in info["detections"]:
            if detection["class_id"] == 0 and detection["track_id"] == 0:  # Ego kart
                ego_bbox = detection
                break
        
        if ego_bbox:
            ego_kart = {
                'kart_name': ego_kart_name,
                'center_x': (ego_bbox['x1'] + ego_bbox['x2']) / 2,
                'center_y': (ego_bbox['y1'] + ego_bbox['y2']) / 2
            }
            kart_objects.append(ego_kart)
            
            # Add other karts with real bounding box data
            for i, kart_name in enumerate(kart_names[1:], 1):
                for detection in info["detections"]:
                    if detection["class_id"] == 0 and detection["track_id"] == i:
                        kart_objects.append({
                            'kart_name': kart_name,
                            'center_x': (detection['x1'] + detection['x2']) / 2,
                            'center_y': (detection['y1'] + detection['y2']) / 2
                        })
                        break
        
        return kart_objects
    except Exception as e:
        print(f"Error extracting kart objects from {info_file}: {e}")
        return []


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate": generate_complete_dataset,
        "generate_focused": generate_focused_dataset,
        "generate_spatial_focused": generate_spatial_focused_dataset
    })


if __name__ == "__main__":
    main()
