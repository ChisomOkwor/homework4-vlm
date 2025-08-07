# Homework 4 - A vision language model for tux

In this homework, we will train (fine-tune) two vision-language models on the SuperTuxKart data [here](https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip).
The first is a generative model, as known the Multimodal Large Language Model (MLLM); the second is a contrastive model, which is a simplified version of the Contrastive Language-Image Model (CLIP).
In the first part, we will focus on the most important aspect of the VLM pipeline: The data-pipeline.
We will use vision labels of the SuperTuxKart dataset to produce question/answer labels for the same set of images.
In the second part, we will focus on building a toy CLIP model and finetune it to do multi-choice question answering.
To fuel this, we will use the the SuperTuxKart dataset to generate some paired image-captions data.



1. Train an VLM (`base_vlm.py`) using `finetune.py`.

2. Build a CLIP model (`clip.py`) based on the components in the VLM in Part 1.
Specifically, We use the vision model in the VLM as the CLIP's vision encoder, and use the LLM in the VLM as the CLIP's text encoder.
Finally, we train the CLIP to do multi-choice question answering.

The starter code contains a minimal training script and dataset.

- `data.py` load a dataset of *images*, *questions*, and *answers* specified in a json file. See `data/train_demo/balanced_qa_pairs.json` for an example.
- `base_vlm.py` sets of a VLM model that can both train and evaluate on the above training data
- `finetune.py` fine-tunes a VLM on a specific dataset
- `clip.py` trains a CLIP on a specific dataset

To get started, familiarize yourself with the starter code, download and unzip the data.

```bash
wget https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip -O supertux_data.zip
unzip supertux_data.zip
```

Then train a model on the demo data we provided

```bash
python -m homework.finetune demo_train
```

and benchmark this model

```bash
python -m homework.finetune test path/to/your/checkpoint
```

Do not expect the model to perform very well, after all it was trained on only 5 question-answer pairs.
Your task is to massively expand this training set.
The checkpoint path needs to include `adapter_config.json` and `adapter_model.safetensors`.

Build a similar data pipeline. Instead of (`question`， `answer`, `image_file`) triplet, CLIP takes as input (`caption`, `image_file`) pairs.

Implement the missing pieces, and you can train a CLIP model using

```bash
python -m homework.clip train
```

and test this model using

```bash
python -m homework.clip test path/to/your/checkpoint
```

The checkpoint path needs to include `adapter_config.json`, `adapter_model.safetensors`, and `additional_weights.pt`.

## Grading

Each part will take 50 pts.
To get 50pts on the first part, you should answer 70% of questions correctly.
The score falls off linearly till 0%.
To get 50pts on the second part, you should answer 70% of questions correctly.
The score falls off linearly till 20% (accuracy of random guess).
There is a 5pt extra credit for submissions reaching 85% accuracy (linearly from 80%).

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

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
    kart_names = info["names"][view_index]

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

        karts.append({
            "instance_id": int(track_id),
            "kart_name": kart_names.get(str(int(track_id)), f"kart_{track_id}"),
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
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    qa_pairs = []

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track = extract_track_info(info_path)

    if not karts:
        return qa_pairs

    ego = next(k for k in karts if k["is_center_kart"])
    ego_id = ego["instance_id"]
    ego_name = ego["kart_name"]
    ego_x, ego_y = ego["center"]

    # 1. What kart is the ego car?
    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego_name
    })

    # 2. How many karts are in the scenario?
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts))
    })

    # 3. What track is this?
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track
    })

    # 4. Relative position questions
    directions = {
        "left": lambda x: x < ego_x,
        "right": lambda x: x > ego_x,
        "front": lambda y: y < ego_y,
        "behind": lambda y: y > ego_y,
    }

    counts = {"left": 0, "right": 0, "front": 0, "behind": 0}

    for k in karts:
        if k["is_center_kart"]:
            continue
        name = k["kart_name"]
        x, y = k["center"]

        # Relative position (e.g., front/behind)
        if directions["front"](y):
            rel_pos = "front"
        elif directions["behind"](y):
            rel_pos = "behind"
        else:
            rel_pos = "side"

        qa_pairs.append({
            "question": f"Is {name} in front of or behind the ego car?",
            "answer": rel_pos
        })

        # Count for directional QA
        for dir_name, func in directions.items():
            if func(x if dir_name in ["left", "right"] else y):
                counts[dir_name] += 1

    # 5. Counting questions
    for dir_name, count in counts.items():
        qa_pairs.append({
            "question": f"How many karts are {dir_name} of the ego car?",
            "answer": str(count)
        })

    return qa_pairs

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
    fire.Fire({"check": check_qa_pairs})


if __name__ == "__main__":
    main()
## Building a VLM data-pipeline (50 pts)

Your main task is to massively expand the dataset for VLM training. Follow the 5 kind of questions given in our demo training set.

`generate_qa.py` provides some initial code to parse and visualize the supertux data.

Run

```bash
python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0
```

The visualize the extracted supertuxkart information and your generated questions.
Finally, write question-answer pairs into a `..._qa_pairs.json` file in `data/train/` and train your model using

```bash
python -m homework.finetune train
```

Do NOT train on the validation data provided.
It will inflate your validation accuracy, and unlikely generalizes to the test set.

## Train a CLIP model (50 pts)

Use the similar data-pipeline in the previous section to generate (`image_file`, `caption`) pairs.

Run

```bash
python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0
```

The visualize the extracted supertuxkart information and your generated captions.
Finally, write captions pairs into a `..._captions.json` file in `data/train/` and train your model using

```bash
python -m homework.clip train
```

## Submission

Once you finished the assignment, create a submission bundle using:

```bash
python3 bundle.py homework [YOUR UT ID]
```

Delete any old checkpoints from your homework directory to keep the model size below 50MB.

Submit the zip file on Canvas. Please note that the maximum file size our grader accepts is **50MB**. Please keep your solution compact.
Please double-check that your zip file was properly created, by grading it again:

```bash
python3 -m grader [YOUR UT ID].zip
```

## Online grader

We will use an automated grader through Canvas to grade all your submissions. There is a soft limit of **5** submissions per assignment. Please contact the course staff before going over this limit, otherwise your submission might be counted as invalid.

The online grading system will use a slightly modified version of Python and the grader:

- Please do not use the `exit` or `sys.exit` command, it will likely lead to a crash in the grader
- Please do not try to access, read, or write files outside the ones specified in the assignment. This again will lead to a crash. File writing is disabled.
- Network access is disabled. Please do not try to communicate with the outside world.
- Forking is not allowed!
- `print` or `sys.stdout.write` statements from your code are ignored and not returned.

Please do not try to break or hack the grader. Doing so will have negative consequences for your standing in this class and the program.

## Installation

We encourage using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to install the required packages.

```bash
conda create --name advances_in_deeplearning python=3.12 -y
conda activate advances_in_deeplearning
```

First, install [PyTorch](https://pytorch.org/get-started/locally/)

Then install additional dependencies:

```bash
pip install -r requirements.txt
```
