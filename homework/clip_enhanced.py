from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
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


def load_clip_model(model_name: str = "clip_enhanced/best_checkpoint"):
    """Load the best enhanced CLIP model for grading"""
    return load(model_name)


def load(model_name: str = "clip_enhanced/best_checkpoint"):
    from pathlib import Path

    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    clip = CLIPEnhanced(vision_encoder, text_encoder)
    clip = PeftModel.from_pretrained(clip, model_path).to(device)

    clip.model.load_pretrained(model_path)
    clip.model.eval()

    return clip


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training.
    """
    # Stack image pixel values
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape

    # Gather and pad text input_ids and attention_mask
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]

    # Pad sequences to the same length
    max_length = max(len(seq) for seq in input_ids)
    padded_input_ids = []
    padded_attention_mask = []

    for i in range(len(input_ids)):
        seq_len = len(input_ids[i])
        pad_length = max_length - seq_len
        
        padded_input_ids.append(torch.cat([input_ids[i], torch.zeros(pad_length, dtype=input_ids[i].dtype)]))
        padded_attention_mask.append(torch.cat([attention_mask[i], torch.zeros(pad_length, dtype=attention_mask[i].dtype)]))

    return {
        "pixel_values": pixel_values,  # Keep original dtype (bfloat16 or float)
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_input_ids),  # placeholder to fit the collator
    }


class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        # Enhanced augmentation for improved model
        self.image_processor = tv.transforms.Compose(
            [
                tv.transforms.Resize(200),  # Slightly larger for random crop
                tv.transforms.RandomResizedCrop(192, scale=(0.6, 1.0)),  # More aggressive crop
                tv.transforms.RandomHorizontalFlip(p=0.5),  # More flipping
                tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Stronger color jitter
                tv.transforms.RandomRotation(10),  # Small rotation
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pair = self.dataset[idx]
        image = Image.open(pair["image_path"]).convert("RGB")
        
        # Process image
        pixel_values = self.image_processor(image)
        # Use bfloat16 for inputs when model uses bfloat16, except on CPU
        if device != "cpu":
            pixel_values = pixel_values.bfloat16()
        else:
            pixel_values = pixel_values.float()

        # Process text (add EOS token)
        text = pair["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=False, truncation=True)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(0).long(),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
        }


class CLIPEnhanced(nn.Module):
    def __init__(
        self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 512, temperature: float = 0.05
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature
        
        # Enhanced projection layers with deeper networks
        # Vision encoder output is 768-dim (avg) + 768-dim (max) = 1536-dim total
        # Text encoder output is 576-dim
        self.vision_projection = nn.Sequential(
            nn.Linear(768 * 2, proj_dim * 2),  # 1536 input for concat avg+max pooling
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim * 2, proj_dim * 2),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(proj_dim * 2, proj_dim)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(576, proj_dim * 2), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim * 2, proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(proj_dim * 2, proj_dim)
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(image)

    def encode_text(self, text: str) -> torch.Tensor:
        return self.text_encoder(text)

    def set_trainable_parameters(self):
        """
        This method sets the trainable parameters of the CLIP model.
        """
        # Freeze the vision and text encoders
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Unfreeze the projection layers
        for param in self.vision_projection.parameters():
            param.requires_grad = True
        for param in self.text_projection.parameters():
            param.requires_grad = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for CLIP model.
        Args:
            pixel_values: Image tensor of shape (batch_size, channels, height, width)
            input_ids: Text input token ids of shape (batch_size, sequence_length)
            attention_mask: Attention mask for text of shape (batch_size, sequence_length)
            labels: Labels for training (not used in CLIP)
            (NOTE: you don't need to use the variable `labels`, this is just for compatibility with the Trainer class)
            (Hint: refer to returned values of the __getitem__ method in the CaptionDatasetForTraining class)
        Returns:
            TODO: think about the what values should be returned
        """
        # Encode images with enhanced pooling
        vision_outputs = self.vision_encoder(pixel_values)
        # Use both global average and max pooling for richer representations
        vision_last_hidden = vision_outputs.last_hidden_state
        vision_avg_pool = vision_last_hidden.mean(dim=1)
        vision_max_pool = vision_last_hidden.max(dim=1)[0]
        vision_features = torch.cat([vision_avg_pool, vision_max_pool], dim=-1)
        
        # Encode text with attention-weighted pooling
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_last_hidden = text_outputs.last_hidden_state
        
        # Use attention mask for proper pooling (ignore padding tokens)
        if attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(text_last_hidden.size()).float()
            text_masked = text_last_hidden * attention_mask_expanded
            text_features = text_masked.sum(dim=1) / (attention_mask_expanded.sum(dim=1) + 1e-8)
        else:
            text_features = text_last_hidden.mean(dim=1)
        
        # Project to common embedding space
        vision_embeddings = self.vision_projection(vision_features)
        text_embeddings = self.text_projection(text_features)
        
        # Normalize embeddings
        vision_embeddings = nn.functional.normalize(vision_embeddings, p=2, dim=1)
        text_embeddings = nn.functional.normalize(text_embeddings, p=2, dim=1)
        
        # Compute similarity matrix and loss for training
        similarity_matrix = torch.matmul(vision_embeddings, text_embeddings.T) / self.temperature
        
        # Create contrastive learning targets (diagonal should be positive pairs)
        batch_size = similarity_matrix.size(0)
        targets = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Contrastive loss (both directions)
        loss_img_to_txt = nn.functional.cross_entropy(similarity_matrix, targets)
        loss_txt_to_img = nn.functional.cross_entropy(similarity_matrix.T, targets)
        total_loss = (loss_img_to_txt + loss_txt_to_img) / 2
        
        return total_loss, vision_embeddings, text_embeddings, self.temperature


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    """
    Get target modules for LoRA fine-tuning.
    """
    target_modules = []
    for name, module in model.named_modules():
        # Include all linear layers in vision/text encoders AND the projection layers
        if isinstance(module, nn.Linear) and (
            "vision_encoder" in name or 
            "text_encoder" in name or 
            "vision_projection" in name or 
            "text_projection" in name
        ):
            target_modules.append(name)
    return target_modules


def evaluate_checkpoint(ckpt_path: str, val_dataset: str = "valid_grader") -> float:
    """
    Evaluate a checkpoint and return accuracy.
    """
    try:
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

        with torch.no_grad():
            for pair in testset:
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
                
                # Get model prediction
                _, vision_feature, text_feature, temperature = clip(pixel_values, input_ids, attention_mask)
                
                similarities = torch.matmul(vision_feature, text_feature.T) / temperature
                prediction = similarities.argmax(dim=-1).item()
                
                if prediction == pair["correct_index"]:
                    correct_count += 1
                total_count += 1
        
        accuracy = correct_count / total_count
        return accuracy
    
    except Exception as e:
        print(f"Error evaluating checkpoint: {e}")
        return 0.0


def train(
    data_dir: Path | None = None,
    output_dir: str = "clip_enhanced",
    num_train_epochs: float = 0.5,  # More epochs for fresh training
    per_device_train_batch_size: int = 128,  # Smaller batch for stability
    gradient_accumulation_steps: int = 8,  # Higher accumulation for effective larger batch
    learning_rate: float = 1e-4,  # Standard learning rate for fresh training
    num_workers: int = 16,
    eval_steps: int = 50,  # Less frequent eval for fresh training
    early_stopping_patience: int = 3,  # More patience for fresh training
    target_accuracy: float = 0.7,  # Target accuracy for early stopping  
    min_improvement: float = 0.01,  # Standard improvement threshold
    baseline_accuracy: float = 0.0,  # No baseline for fresh training
):
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    # Create enhanced model with improved architecture
    model = CLIPEnhanced(vision_encoder, text_encoder).to(device).bfloat16()
    model.set_trainable_parameters()

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=16,  # Higher rank for enhanced model
        lora_alpha=32,
        lora_dropout=0.1,  # Some dropout for regularization
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Load datasets
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    
    train_dataset = CaptionDatasetForTraining(CaptionDataset(data_dir / "train"), processor)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        bf16=True,
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        report_to="tensorboard",
        load_best_model_at_end=False,  # We handle this manually
        metric_for_best_model="accuracy", 
        greater_is_better=True,
        save_total_limit=3,
    )

    # Enhanced evaluation callback
    class EvaluationCallback:
        def __init__(self, output_dir, target_accuracy, patience, min_improvement, baseline_accuracy):
            self.output_dir = Path(output_dir)
            self.target_accuracy = target_accuracy
            self.patience = patience
            self.min_improvement = min_improvement
            self.baseline_accuracy = baseline_accuracy
            self.best_accuracy = baseline_accuracy
            self.patience_counter = 0

        def on_step_end(self, trainer):
            if trainer.state.global_step % trainer.args.eval_steps == 0:
                print(f"\n=== Evaluation at step {trainer.state.global_step} ===")
                
                # Save current model for evaluation
                temp_dir = self.output_dir / f"temp_checkpoint_{trainer.state.global_step}"
                trainer.save_model(temp_dir)
                trainer.model.model.save_pretrained(temp_dir)
                
                # Evaluate accuracy
                accuracy = evaluate_checkpoint(str(temp_dir))
                print(f"Current accuracy: {accuracy:.4f}")
                
                # Enhanced stopping logic for fresh training
                improvement = accuracy - self.best_accuracy
                
                if accuracy >= self.target_accuracy:
                    print(f"🎯 TARGET ACCURACY REACHED: {accuracy:.4f}!")
                    best_dir = self.output_dir / "best_checkpoint"
                    trainer.save_model(best_dir)
                    trainer.model.model.save_pretrained(best_dir)
                    trainer.should_training_stop = True
                    return
                
                if improvement >= self.min_improvement:
                    self.best_accuracy = accuracy
                    self.patience_counter = 0
                    print(f"🎉 NEW BEST ACCURACY: {accuracy:.4f} (improvement: +{improvement:.4f})")
                    
                    # Update best_checkpoint
                    best_dir = self.output_dir / "best_checkpoint"
                    trainer.save_model(best_dir)
                    trainer.model.model.save_pretrained(best_dir)
                    print(f"✅ Updated best_checkpoint with {accuracy:.4f} accuracy")
                else:
                    self.patience_counter += 1
                    print(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
                    
                    if self.patience_counter >= self.patience:
                        print("Early stopping triggered!")
                        trainer.should_training_stop = True
                
                # Clean up temp checkpoint
                import shutil
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

    # Create trainer with enhanced callback
    evaluation_callback = EvaluationCallback(
        output_dir, target_accuracy, early_stopping_patience, min_improvement, baseline_accuracy
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
    )

    # Add evaluation callback manually
    trainer.evaluation_callback = evaluation_callback
    original_on_step_end = trainer.callback_handler.on_step_end
    
    def enhanced_on_step_end(args, state, control, model=None, **kwargs):
        result = original_on_step_end(args, state, control, model=model, **kwargs)
        evaluation_callback.on_step_end(trainer)
        return result
    
    trainer.callback_handler.on_step_end = enhanced_on_step_end

    print("Starting enhanced CLIP training...")
    trainer.train()

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_accuracy = evaluate_checkpoint(str(output_dir / "best_checkpoint"))
    print(f"Final best accuracy: {final_accuracy:.4f}")

    return model, final_accuracy