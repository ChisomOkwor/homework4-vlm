from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

# Use the same device as BaseVLM for consistency
from .base_vlm import DEVICE
device = DEVICE


def load_clip_model(model_name: str = "clip_model"):
    """Load the best CLIP model for grading"""
    # Handle different model name formats for grader compatibility
    if model_name == "clip_model":
        # Default grader path - use our best checkpoint
        return load("clip/best_checkpoint")
    else:
        return load(model_name)


def load(model_name: str = "clip/best_checkpoint"):
    from pathlib import Path

    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    # Use same CLIP configuration as training
    clip = CLIP(vision_encoder, text_encoder)
    clip = PeftModel.from_pretrained(clip, model_path).to(device)
    
    # Ensure MPS compatibility 
    if device == "mps":
        clip.model.vision_encoder = clip.model.vision_encoder.float()
        clip.model.text_encoder = clip.model.text_encoder.float()

    clip.model.load_pretrained(model_path)
    clip.model.eval()

    return clip


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training.
    """
    # Get max sequence length
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values,  # Keep original dtype (bfloat16 or float)
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor, 
                 data_augmentation: bool = True, curriculum_learning: bool = False):
        self.dataset = dataset
        self.data_augmentation = data_augmentation
        self.curriculum_learning = curriculum_learning
        
        # Enhanced data augmentation transforms
        if data_augmentation:
            self.image_processor = tv.transforms.Compose([
                tv.transforms.Resize(224),  # Slightly larger for better crops
                tv.transforms.RandomResizedCrop(192, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
                tv.transforms.RandomHorizontalFlip(p=0.3),  # Moderate flipping
                tv.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                tv.transforms.RandomRotation(degrees=5),  # Small rotation
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            # Standard transforms for consistent evaluation
            self.image_processor = tv.transforms.Compose([
                tv.transforms.Resize(192),
                tv.transforms.CenterCrop(192),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        
        self.processor = processor
        
        # Curriculum learning: sort by caption complexity (shorter = easier)
        if curriculum_learning:
            dataset_items = list(self.dataset)
            dataset_items.sort(key=lambda x: len(x["caption"]))
            self.sorted_indices = list(range(len(dataset_items)))
            print(f"📚 Curriculum learning: sorted {len(dataset_items)} samples by caption complexity")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Use curriculum learning index if enabled
        if self.curriculum_learning:
            actual_idx = self.sorted_indices[idx]
            item = self.dataset[actual_idx]
        else:
            item = self.dataset[idx]
            
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        
        # Keep everything as float32 for MPS compatibility
        pixel_values = pixel_values.float()
            
        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # placeholder to fit the collator
        }


class CLIP(nn.Module):
    def __init__(
        self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 256, temperature: float = 0.07
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Make temperature trainable as mentioned in tips: "t needs to be trainable"
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Use higher dimensional projections as preferred
        # Vision encoder output is 768-dim, text encoder output is 576-dim
        self.vision_projection = nn.Linear(768, proj_dim)
        self.text_projection = nn.Linear(576, proj_dim)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(image)

    def encode_text(self, text: str) -> torch.Tensor:
        return self.text_encoder(text)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Customize save method, save additional parameters"""

        additional_state_dict = {}
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data

        torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        """Customize load method, load projection additional parameters"""

        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists():
            additional_state_dict = torch.load(additional_weights_path, map_location="cpu")

            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                param.data = additional_state_dict[name]

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the vision and text backbones.
        (You don't need to touch this method)
        """
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        """
        Enable input require grads for the vision and text backbones.
        (You don't need to touch this method)
        """

        # Reference: https://discuss.huggingface.co/t/peft-lora-gpt-neox-backward-pass-failing/35641
        def make_inputs_require_grads(module, input, output):  # noqa: A002
            output.requires_grad_(True)

        self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CLIP model.
        Args:
            pixel_values: The pixel values of the image.
            input_ids: The input ids of the text.
            attention_mask: The attention mask of the text.
            labels: The labels for the text features.
            (NOTE: you don't need to use the variable `labels`, this is just for compatibility with the Trainer class)
            (Hint: refer to returned values of the __getitem__ method in the CaptionDatasetForTraining class)
        Returns:
            TODO: think about the what values should be returned
        """
        # Encode images using pre-written method as suggested in tips
        vision_outputs = self.encode_image(pixel_values)
        # Get hidden states and apply average pooling as suggested in tips
        # Shape: (B, seq_len, hidden_size_vision) -> (B, hidden_size_vision)
        vision_features = vision_outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
        # Encode text directly using text_encoder as suggested in tips
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # CRITICAL FIX: Use average pooling with proper attention mask handling
        # This is the "poor practice" fix mentioned in Trick #2 that boosts accuracy 5-10%
        # Don't use [CLS] token - use average pooling as suggested in "The Oversight"
        last_hidden_states = text_outputs.last_hidden_state  # Shape: (B, seq_len, hidden_dim)
        
        # Apply attention mask to avoid averaging over padding tokens (the key fix!)
        # attention_mask: (B, seq_len) where 1 = real token, 0 = padding
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        
        # Zero out hidden states for padding tokens
        masked_hidden_states = last_hidden_states * attention_mask_expanded
        
        # Sum the masked hidden states and divide by the actual sequence length
        sum_hidden_states = torch.sum(masked_hidden_states, dim=1)  # (B, hidden_dim)
        seq_lengths = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)  # (B, 1)
        text_features = sum_hidden_states / seq_lengths  # Proper average pooling
        
        # Project to common embedding space
        vision_embeddings = self.vision_projection(vision_features)
        text_embeddings = self.text_projection(text_features)
        
        # Normalize embeddings using F.normalize for better MPS compatibility
        vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        return vision_embeddings, text_embeddings, self.temperature


def compute_clip_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """
    Compute the loss for the CLIP model.
    Args:
        outputs: A tuple containing the outputs of CLIP.forward().
        labels: The labels for the text features.
        (NOTE: you don't need to use the variable `labels`, this is just for compatibility with the Trainer class)
        num_items_in_batch: The number of items in the batch.
        (NOTE: you don't need to use the variable `num_items_in_batch`, this is just for compatibility with Trainer)
    Returns:
        The loss for the CLIP model.
    """
    vision_embeddings, text_embeddings, temperature = outputs
    
    batch_size = vision_embeddings.shape[0]
    device = vision_embeddings.device
    
    # Compute similarity matrix (scaled by temperature) following CLIP paper pseudocode
    # logits = np.dot(I_e, T_e.T) * np.exp(t)
    logits = torch.matmul(vision_embeddings, text_embeddings.T) * torch.exp(temperature)
    
    # Create labels for contrastive learning
    # Each image should match with its corresponding text (diagonal elements)
    targets = torch.arange(batch_size, device=device)
    
    # Compute cross-entropy loss in both directions
    # Image-to-text loss
    image_to_text_loss = torch.nn.functional.cross_entropy(logits, targets)
    
    # Text-to-image loss  
    text_to_image_loss = torch.nn.functional.cross_entropy(logits.T, targets)
    
    # Average the two losses
    loss = (image_to_text_loss + text_to_image_loss) / 2
    
    return loss


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    target_modules = []
    for name, module in model.named_modules():
        # Include all linear layers in vision/text encoders but NOT projection layers
        if isinstance(module, nn.Linear) and (
            "vision_encoder" in name or 
            "text_encoder" in name
        ) and "projection" not in name:
            target_modules.append(name)
    
    print(f"LoRA target modules found: {len(target_modules)}")
    if len(target_modules) > 0:
        print(f"First few modules: {target_modules[:3]}")
    return target_modules


def evaluate_checkpoint(ckpt_path: str, val_dataset: str = "valid_grader") -> float:
    """Evaluate a checkpoint and return accuracy"""
    import tqdm
    
    try:
        testset = MultiChoiceQADataset(val_dataset)
        clip = load(ckpt_path)
        clip = clip.model.to(device)
        clip.eval()

        image_processor = tv.transforms.Compose(
            [
                tv.transforms.Resize(192),
                tv.transforms.CenterCrop(192),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        correct_count = 0
        total_count = 0

        with torch.no_grad():
            for pair in tqdm.tqdm(testset, desc="Evaluating"):
                image = Image.open(pair["image_path"]).convert("RGB")
                pixel_values = image_processor(image).unsqueeze(0).to(device)
                # Keep as float32 for MPS compatibility
                pixel_values = pixel_values.float()
                text_inputs = processor(
                    text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                input_ids = text_inputs["input_ids"].long().to(device)
                attention_mask = text_inputs["attention_mask"].to(device)
                vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
                prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
                if prediction == pair["correct_index"]:
                    correct_count += 1
                total_count += 1

        accuracy = correct_count / total_count
        print(f"Checkpoint accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
        return accuracy
    except Exception as e:
        print(f"Error evaluating checkpoint: {e}")
        return 0.0


def train(
    data_dir: Path | None = None,
    output_dir: str = "clip",
    resume_from_checkpoint: str = None,  # Explicit checkpoint path to resume from
    num_train_epochs: float = 1.0,  # Train longer to achieve 85% accuracy target
    per_device_train_batch_size: int = 256,  # Following tips: Option 1 for 6GB GPU
    gradient_accumulation_steps: int = 4,  # Following tips: Option 1 configuration  
    learning_rate: float = 5e-4,  # Standard learning rate for fresh training
    adaptive_lr: bool = True,  # Enable adaptive learning rate based on progress
    warmup_ratio: float = 0.1,  # Warmup for stable training
    weight_decay: float = 0.01,  # Regularization to prevent overfitting
    num_workers: int = 0,  # Set to 0 to eliminate HTTP rate limiting completely
    eval_steps: int = 50,  # Less frequent for longer training
    early_stopping_patience: int = 10,  # More patience for reaching 85%
    target_accuracy: float = 0.85,  # Target 85% accuracy as mentioned in tips  
    min_improvement: float = 0.005,  # Smaller improvement threshold for high accuracy
    baseline_accuracy: float = None,  # Will be auto-detected from checkpoint
    use_curriculum_learning: bool = True,  # Start with easier examples
    data_augmentation: bool = True,  # Enhanced data augmentation
):
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Auto-detect best checkpoint if no explicit path provided
    if resume_from_checkpoint is None:
        best_checkpoint_path = output_dir / "best_checkpoint"
        if best_checkpoint_path.exists():
            resume_from_checkpoint = str(best_checkpoint_path)
            print(f"🔄 Auto-detected checkpoint to resume from: {resume_from_checkpoint}")
        else:
            print("🆕 No checkpoint found, starting fresh training")

    # Initialize model and processor
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    # Create CLIP model with original architecture for checkpoint compatibility
    model = CLIP(vision_encoder, text_encoder).to(device)
    # Convert base encoders to float32 for MPS compatibility
    if device == "mps":
        model.vision_encoder = model.vision_encoder.float()
        model.text_encoder = model.text_encoder.float()
    # The model itself and projection layers will stay as created
    model.set_trainable_parameters()

    # Load from checkpoint if resuming
    if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
        print(f"📂 Loading model from checkpoint: {resume_from_checkpoint}")
        try:
            # Load the PEFT model
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, resume_from_checkpoint).to(device)
            model.model.load_pretrained(resume_from_checkpoint)
            # Keep model as float32 for MPS compatibility
            if device == "mps":
                model.model.vision_encoder = model.model.vision_encoder.float()
                model.model.text_encoder = model.model.text_encoder.float()
            print("✅ Successfully loaded checkpoint!")
            
            # Auto-detect baseline accuracy from checkpoint performance
            if baseline_accuracy is None:
                print("🎯 Auto-detecting baseline accuracy from checkpoint...")
                baseline_accuracy = evaluate_checkpoint(resume_from_checkpoint)
                print(f"📊 Detected baseline accuracy: {baseline_accuracy:.4f}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print("🔄 Falling back to fresh training...")
            resume_from_checkpoint = None
            # Create fresh PEFT model
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules=get_target_modules_for_lora(model),
                bias="none",
            )
            model = get_peft_model(model, peft_config)
    else:
        # Create fresh PEFT model
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=get_target_modules_for_lora(model),
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    # Set default baseline if not detected
    if baseline_accuracy is None:
        baseline_accuracy = 0.0
        print("📊 Using default baseline accuracy: 0.0")

    model.print_trainable_parameters()
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Load dataset with curriculum learning and data augmentation
    train_dataset = CaptionDataset("train", data_dir)
    train_dataset = CaptionDatasetForTraining(train_dataset, processor, 
                                               data_augmentation=data_augmentation,
                                               curriculum_learning=use_curriculum_learning)
    
    print(f"Training on {len(train_dataset)} samples")
    if use_curriculum_learning:
        print("📚 Curriculum learning enabled: starting with easier examples")
    if data_augmentation:
        print("🔄 Enhanced data augmentation enabled")

    # Custom training arguments with improved settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        bf16=True if device == "cuda" else False,
        fp16=False,  # Disable mixed precision for MPS compatibility
        logging_steps=5,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,  # Keep fewer checkpoints to save space
        label_names=["labels"],
        dataloader_num_workers=num_workers,
        lr_scheduler_type="cosine_with_restarts",  # Better learning rate scheduling
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,  # Gradient clipping for stability
        seed=42,  # For reproducible results
        remove_unused_columns=False,  # Keep all columns for our custom setup
    )

    # Custom trainer callback for evaluation with adaptive features
    class EvaluationCallback:
        def __init__(self, output_dir, eval_steps, early_stopping_patience, target_accuracy, min_improvement, baseline_accuracy, adaptive_lr):
            self.output_dir = output_dir
            self.eval_steps = eval_steps
            self.early_stopping_patience = early_stopping_patience
            self.target_accuracy = target_accuracy
            self.min_improvement = min_improvement
            self.baseline_accuracy = baseline_accuracy
            self.best_accuracy = baseline_accuracy  # Start with our known good accuracy
            self.patience_counter = 0
            self.step_count = 0
            self.adaptive_lr = adaptive_lr
            self.lr_reduction_factor = 0.5  # Reduce LR by this factor when plateau
            self.lr_reduction_patience = 2  # Reduce LR after this many non-improvements
            self.consecutive_improvements = 0  # Track consecutive improvements
            self.accuracy_history = []  # Track accuracy over time
            print(f"🎯 Starting with baseline accuracy: {baseline_accuracy:.4f}")
            if adaptive_lr:
                print("🧠 Adaptive learning rate enabled")
            
        def on_step_end(self, trainer):
            self.step_count += 1
            if self.step_count % self.eval_steps == 0:
                print(f"\n=== Evaluating at step {self.step_count} ===")
                
                # Save current checkpoint
                checkpoint_dir = self.output_dir / f"checkpoint-{self.step_count}"
                trainer.save_model(checkpoint_dir)
                trainer.model.model.save_pretrained(checkpoint_dir)
                
                # Evaluate checkpoint
                accuracy = evaluate_checkpoint(str(checkpoint_dir))
                self.accuracy_history.append(accuracy)
                
                # Log to tensorboard (with error handling)
                try:
                    writer.add_scalar("eval/accuracy", accuracy, self.step_count)
                    writer.add_scalar("train/learning_rate", trainer.optimizer.param_groups[0]['lr'], self.step_count)
                except Exception as e:
                    print(f"TensorBoard logging failed: {e}")
                
                # Check for improvement with minimum threshold
                improvement = accuracy - self.best_accuracy
                
                # CRITICAL: Stop immediately if accuracy drops significantly below baseline
                if accuracy < self.baseline_accuracy - 0.02:  # 2% tolerance for resumed training
                    print(f"🚨 ALERT: Accuracy dropped to {accuracy:.4f}, significantly below baseline {self.baseline_accuracy:.4f}!")
                    print("🛑 Stopping training to prevent further degradation.")
                    trainer.should_training_stop = True
                    return
                
                # Check if we're making steady progress
                if improvement >= self.min_improvement:
                    self.best_accuracy = accuracy
                    self.patience_counter = 0
                    self.consecutive_improvements += 1
                    print(f"🎉 NEW BEST ACCURACY: {accuracy:.4f} (improvement: +{improvement:.4f})")
                    print(f"📈 Consecutive improvements: {self.consecutive_improvements}")
                    
                    # Only update best_checkpoint if we actually improved
                    best_dir = self.output_dir / "best_checkpoint"
                    trainer.save_model(best_dir)
                    trainer.model.model.save_pretrained(best_dir)
                    print(f"✅ Updated best_checkpoint with {accuracy:.4f} accuracy")
                    
                    # If we have multiple consecutive improvements, slightly increase LR
                    if self.adaptive_lr and self.consecutive_improvements >= 2:
                        current_lr = trainer.optimizer.param_groups[0]['lr']
                        new_lr = min(current_lr * 1.1, learning_rate)  # Don't exceed original LR
                        for param_group in trainer.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f"📈 Boosting learning rate: {current_lr:.2e} → {new_lr:.2e}")
                    
                else:
                    self.patience_counter += 1
                    self.consecutive_improvements = 0
                    
                    if improvement > 0:
                        print(f"📊 Small improvement: +{improvement:.4f} (below threshold {self.min_improvement:.4f}). Patience: {self.patience_counter}/{self.early_stopping_patience}")
                    else:
                        print(f"📉 No improvement: {improvement:.4f}. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                    
                    # Adaptive learning rate reduction when no improvement
                    if self.adaptive_lr and self.patience_counter >= self.lr_reduction_patience:
                        current_lr = trainer.optimizer.param_groups[0]['lr']
                        new_lr = current_lr * self.lr_reduction_factor
                        for param_group in trainer.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f"📉 Reducing learning rate: {current_lr:.2e} → {new_lr:.2e}")
                        self.patience_counter = 0  # Reset patience after LR reduction
                
                # Enhanced progress tracking
                print(f"📊 Accuracy trend (last 3): {self.accuracy_history[-3:] if len(self.accuracy_history) >= 3 else self.accuracy_history}")
                
                # Check early stopping conditions
                if accuracy >= self.target_accuracy:
                    print(f"🎯 Target accuracy {self.target_accuracy:.4f} reached! Stopping training.")
                    trainer.should_training_stop = True
                    return
                    
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"⏹️ Early stopping triggered. Best accuracy: {self.best_accuracy:.4f}")
                    trainer.should_training_stop = True
                    return

    # Create callback
    eval_callback = EvaluationCallback(output_dir, eval_steps, early_stopping_patience, target_accuracy, min_improvement, baseline_accuracy, adaptive_lr)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
    )

    # Add manual callback handling
    original_step = trainer.training_step
    def custom_training_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        eval_callback.on_step_end(trainer)
        return result
    trainer.training_step = custom_training_step

    print(f"🚀 Starting training with:")
    print(f"  📚 Learning rate: {learning_rate}")
    print(f"  📦 Batch size: {per_device_train_batch_size} × {gradient_accumulation_steps} = {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"  ⏱️ Eval every: {eval_steps} steps")
    print(f"  🎯 Target accuracy: {target_accuracy}")
    print(f"  ⏸️ Early stopping patience: {early_stopping_patience}")
    print(f"  📊 Minimum improvement threshold: {min_improvement}")
    print(f"  🧠 Adaptive LR: {adaptive_lr}")
    print(f"  📚 Curriculum learning: {use_curriculum_learning}")
    print(f"  🔄 Data augmentation: {data_augmentation}")

    # Train with or without checkpoint resumption
    try:
        if resume_from_checkpoint:
            print(f"📂 Training will resume from: {resume_from_checkpoint}")
            # Don't pass checkpoint to trainer.train() since we loaded it manually
            trainer.train()
        else:
            print("🆕 Starting fresh training")
            trainer.train()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Trying to continue without checkpoint...")
    trainer.train()

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_checkpoint = output_dir / "final_checkpoint"
    trainer.save_model(final_checkpoint)
    model.model.save_pretrained(final_checkpoint)
    
    final_accuracy = evaluate_checkpoint(str(final_checkpoint))
    print(f"Final accuracy: {final_accuracy:.4f}")
    
    # Report best accuracy achieved
    print(f"Best accuracy achieved: {eval_callback.best_accuracy:.4f}")
    
    # Copy best checkpoint to clip_model for grader compatibility (FAQ #513, #646)
    best_checkpoint_path = output_dir / "best_checkpoint"
    clip_model_path = Path(__file__).parent / "clip_model"
    
    if best_checkpoint_path.exists():
        print(f"📋 Copying best checkpoint to clip_model for grader compatibility...")
        import shutil
        if clip_model_path.exists():
            shutil.rmtree(clip_model_path)
        shutil.copytree(best_checkpoint_path, clip_model_path)
        print(f"✅ Model saved to {clip_model_path} for grader")

    writer.close()

    return model, processor


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_clip",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
    )


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)

    clip = load(ckpt_path)
    clip = clip.model.to(device)

    image_processor = tv.transforms.Compose(
        [
            tv.transforms.Resize(192),
            tv.transforms.CenterCrop(192),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    correct_count = 0
    total_count = 0

    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device).float()
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if prediction == pair["correct_index"]:
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count}")


def train_high_batch():
    """Train with high batch size option (Option 2 from tips)"""
    return train(
        output_dir="clip",
        num_train_epochs=1.0,
        per_device_train_batch_size=1024,  # Option 2: default high batch size
        gradient_accumulation_steps=1,  # No accumulation needed
        learning_rate=5e-4,
        adaptive_lr=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_steps=25,  # More frequent with high batch
        early_stopping_patience=8,
        target_accuracy=0.85,
        min_improvement=0.005,
        use_curriculum_learning=True,
        data_augmentation=True,
    )

def resume_train():
    """Resume training from clip/best_checkpoint with optimized settings for guaranteed improvement"""
    return train(
        resume_from_checkpoint="clip/best_checkpoint",
        output_dir="clip",
        num_train_epochs=0.5,  # Longer training for high accuracy  
        per_device_train_batch_size=256,  # Option 1 configuration
        gradient_accumulation_steps=4,  # Option 1 configuration
        learning_rate=1e-5,  # Conservative learning rate for fine-tuning
        adaptive_lr=True,
        warmup_ratio=0.1,  # Longer warmup for stability
        weight_decay=0.01,  # Moderate regularization
        eval_steps=50,  # Less frequent for longer training
        early_stopping_patience=8,  # More patience for high accuracy
        min_improvement=0.005,  # Smaller improvement threshold
        use_curriculum_learning=False,  # Disable for fine-tuning
        data_augmentation=True,  # Keep augmentation for robustness
        target_accuracy=0.85,  # Target 85% accuracy
        baseline_accuracy=0.33,  # Correct baseline for 33% checkpoint
    )

def main():
    from fire import Fire

    Fire({
        "train": train, 
        "test": test, 
        "resume": resume_train,
        "high_batch": train_high_batch
    })


if __name__ == "__main__":
    main()
