from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_name: str = "clip"):
    from pathlib import Path

    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    clip = CLIP(vision_encoder, text_encoder)
    clip = PeftModel.from_pretrained(clip, str(model_path)).to(device)

    clip.model.load_pretrained(str(model_path))
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
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        self.image_processor = tv.transforms.Compose(
            [
                tv.transforms.Resize(192),
                tv.transforms.RandomResizedCrop(192, scale=(0.5, 1.0)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        
        # Fix tokenization: avoid double EOS tokens
        # Use add_special_tokens=False to prevent tokenizer from adding its own EOS
        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
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
        self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 128, temperature: float = 0.07
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.proj_dim = proj_dim
        
        # Get the hidden sizes from the encoders
        vision_width = vision_encoder.config.hidden_size
        text_width = text_encoder.config.hidden_size
        
        # Projection layers
        self.image_proj = nn.Linear(vision_width, proj_dim)
        self.text_proj = nn.Linear(text_width, proj_dim)
        
        # Learnable temperature parameter 
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

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
            img_feat: L2-normalized image features
            txt_feat: L2-normalized text features  
            logits: similarity logits matrix
        """
        # Encode images - get vision model outputs
        vision_outputs = self.vision_encoder(pixel_values)
        # Mean pooling over patch tokens (exclude CLS token if present)
        if hasattr(vision_outputs, 'last_hidden_state'):
            # For vision transformers, average pool over spatial dimensions
            vision_features = vision_outputs.last_hidden_state.mean(dim=1)  # [B, hidden_size]
        else:
            vision_features = vision_outputs.pooler_output
        
        # Encode text
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Text pooling: find first true EOS token (not padding), fallback to masked mean
        batch_size, seq_len = input_ids.shape
        text_features = []
        
        for i in range(batch_size):
            # Find first EOS token that's not padding (attention_mask=1)
            eos_token_id = processor.tokenizer.eos_token_id
            valid_positions = attention_mask[i] == 1
            eos_positions = (input_ids[i] == eos_token_id) & valid_positions
            
            if eos_positions.any():
                # Use the first true EOS position
                eos_idx = eos_positions.nonzero(as_tuple=True)[0][0]
                text_feat = text_outputs.last_hidden_state[i, eos_idx]
            else:
                # Fallback: masked mean over valid tokens
                valid_tokens = text_outputs.last_hidden_state[i] * attention_mask[i].unsqueeze(-1)
                text_feat = valid_tokens.sum(dim=0) / attention_mask[i].sum()
            
            text_features.append(text_feat)
        
        text_features = torch.stack(text_features)  # [B, hidden_size]
        
        # Project to shared embedding space
        img_proj = self.image_proj(vision_features)  # [B, proj_dim]
        txt_proj = self.text_proj(text_features)     # [B, proj_dim]
        
        # L2 normalize projected features
        img_feat = torch.nn.functional.normalize(img_proj, p=2, dim=-1)
        txt_feat = torch.nn.functional.normalize(txt_proj, p=2, dim=-1)
        
        # Compute similarity logits with learnable temperature
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(img_feat, txt_feat.T)
        
        return img_feat, txt_feat, logits


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
    img_feat, txt_feat, logits = outputs
    batch_size = logits.shape[0]
    
    # Create labels for contrastive learning (diagonal: image i matches text i)
    target_labels = torch.arange(batch_size, device=logits.device)
    
    # Symmetric InfoNCE loss:
    # - Image-to-text: CE(logits, labels) 
    # - Text-to-image: CE(logits.T, labels)
    # Average the two losses
    img_to_text_loss = torch.nn.functional.cross_entropy(logits, target_labels)
    text_to_img_loss = torch.nn.functional.cross_entropy(logits.T, target_labels)
    
    clip_loss = (img_to_text_loss + text_to_img_loss) / 2
    
    return clip_loss


def compute_validation_accuracy(model, val_dataset: str = "valid_grader") -> float:
    """
    Compute validation accuracy for the CLIP model.
    """
    import tqdm
    
    testset = MultiChoiceQADataset(val_dataset)
    
    # Apply same precision fix for MPS compatibility
    if device == "mps":
        model = model.float()
    
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
    
    model.eval()
    with torch.no_grad():
        for pair in tqdm.tqdm(testset, desc="Validating"):
            image = Image.open(pair["image_path"]).convert("RGB")
            pixel_values = image_processor(image).unsqueeze(0).to(device)
            # Only use bfloat16 for CUDA, not for MPS
            if device == "cuda":
                pixel_values = pixel_values.bfloat16()
            text_inputs = processor(
                text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
            input_ids = text_inputs["input_ids"].long().to(device)
            attention_mask = text_inputs["attention_mask"].to(device)
            vision_feature, text_feature, _ = model(pixel_values, input_ids, attention_mask)
            prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
            if prediction == pair["correct_index"]:
                correct_count += 1
            total_count += 1
    
    accuracy = correct_count / total_count
    model.train()  # Set back to training mode
    return accuracy


class ValidationCallback(TrainerCallback):
    """
    Custom callback to run validation at each checkpoint and save the best model.
    """
    def __init__(self, output_dir: Path, writer: SummaryWriter):
        self.output_dir = output_dir
        self.writer = writer
        self.best_accuracy = 0.0
        self.best_checkpoint_dir = output_dir / "best_checkpoint"
        self.best_checkpoint_dir.mkdir(exist_ok=True)
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Called when a checkpoint is saved."""
        if model is not None:
            print(f"\n🔍 Running validation at step {state.global_step}...")
            
            # Compute validation accuracy
            accuracy = compute_validation_accuracy(model.model)
            print(f"📊 Validation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Log to TensorBoard
            self.writer.add_scalar("validation/accuracy", accuracy, state.global_step)
            
            # Save best model if accuracy improved
            if accuracy > self.best_accuracy:
                print(f"🎉 New best accuracy! {self.best_accuracy:.4f} → {accuracy:.4f}")
                self.best_accuracy = accuracy
                
                # Save the best model
                print(f"💾 Saving best model to {self.best_checkpoint_dir}")
                model.save_pretrained(self.best_checkpoint_dir)
                model.model.save_pretrained(self.best_checkpoint_dir)
                
                # Save accuracy info
                with open(self.best_checkpoint_dir / "best_accuracy.txt", "w") as f:
                    f.write(f"Best accuracy: {accuracy:.6f}\n")
                    f.write(f"Step: {state.global_step}\n")
                    f.write(f"Epoch: {state.epoch:.2f}\n")
                
                print(f"✅ Best model saved!")
            else:
                print(f"📈 Best accuracy remains: {self.best_accuracy:.4f}")
            
            # Log best accuracy to TensorBoard
            self.writer.add_scalar("validation/best_accuracy", self.best_accuracy, state.global_step)


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    target_modules = []
    for name, module in model.named_modules():
        # Only target linear layers in encoders, exclude our projection heads
        if (
            isinstance(module, nn.Linear)
            and ("vision_encoder" in name or "text_encoder" in name)
            and "projection" not in name
            and "image_proj" not in name  # Exclude our projection layers
            and "text_proj" not in name   # Exclude our projection layers
        ):
            target_modules.append(name)

    return target_modules


def train(
    data_dir: Path | None = None,
    output_dir: str = "clip",
    num_train_epochs: float = 1,
    per_device_train_batch_size: int = 32,  # Optimized for macOS MPS
    gradient_accumulation_steps: int = 8,   # Effective batch size 256
    learning_rate: float = 1.25e-4,         # Scaled from 5e-4 × 256/1024
    num_workers: int = 0,                   # Avoid MPS dataloader stalls
    resume_from_checkpoint: str | None = None,  # Path to checkpoint to resume from
):
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and processor
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    model = CLIP(vision_encoder, text_encoder).to(device)
    # Only use bfloat16 for CUDA, not for MPS
    # For MPS, convert the entire model to float32 for compatibility
    if device == "cuda":
        model = model.bfloat16()
    elif device == "mps":
        model = model.float()  # Ensure consistent float32 on MPS
    model.set_trainable_parameters()

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        # target_modules="all-linear",
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # load dataset
    train_dataset = CaptionDataset("train", data_dir)
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        bf16=True if device == "cuda" else False,
        fp16=False,  # Disable fp16 for MPS compatibility
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=4,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
        warmup_steps=500,  # Add warmup steps as recommended
        weight_decay=0.01,  # Add weight decay
        adam_beta1=0.9,     # AdamW beta1
        adam_beta2=0.98,    # AdamW beta2 as recommended
    )

    # Create validation callback and load previous best accuracy if resuming
    validation_callback = ValidationCallback(output_dir, writer)
    
    # If resuming, try to load the previous best accuracy
    if resume_from_checkpoint:
        best_accuracy_file = validation_callback.best_checkpoint_dir / "best_accuracy.txt"
        if best_accuracy_file.exists():
            try:
                with open(best_accuracy_file, "r") as f:
                    first_line = f.readline().strip()
                    if first_line.startswith("Best accuracy:"):
                        validation_callback.best_accuracy = float(first_line.split(":")[1].strip())
                        print(f"📚 Loaded previous best accuracy: {validation_callback.best_accuracy:.4f}")
            except Exception as e:
                print(f"⚠️ Could not load previous best accuracy: {e}")
        else:
            print(f"ℹ️ No previous best accuracy found, starting fresh")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
        callbacks=[validation_callback],
    )

    # Start training (with optional checkpoint resuming)
    if resume_from_checkpoint:
        print(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print(f"🚀 Starting training from scratch")
        trainer.train()

    # save model
    trainer.save_model(output_dir)
    model.model.save_pretrained(output_dir)
    
    # Final validation run
    print(f"\n🏁 Training completed! Running final validation...")
    final_accuracy = compute_validation_accuracy(model.model)
    print(f"📊 Final validation accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    # Update best model if final is better
    if final_accuracy > validation_callback.best_accuracy:
        print(f"🎉 Final model is the best! Updating best checkpoint...")
        validation_callback.best_accuracy = final_accuracy
        model.save_pretrained(validation_callback.best_checkpoint_dir)
        model.model.save_pretrained(validation_callback.best_checkpoint_dir)
        with open(validation_callback.best_checkpoint_dir / "best_accuracy.txt", "w") as f:
            f.write(f"Best accuracy: {final_accuracy:.6f}\n")
            f.write(f"Step: final\n")
            f.write(f"Epoch: {trainer.state.epoch:.2f}\n")
    
    # Training summary
    print(f"\n📈 Training Summary:")
    print(f"   Best validation accuracy: {validation_callback.best_accuracy:.4f} ({validation_callback.best_accuracy*100:.2f}%)")
    print(f"   Final validation accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"   Best model saved to: {validation_callback.best_checkpoint_dir}")
    print(f"   Final model saved to: {output_dir}")

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
    # Apply same precision fix for MPS compatibility
    if device == "mps":
        clip = clip.float()

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
        pixel_values = image_processor(image).unsqueeze(0).to(device)
        # Only use bfloat16 for CUDA, not for MPS
        if device == "cuda":
            pixel_values = pixel_values.bfloat16()
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if prediction == pair["correct_index"]:
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count}")


def list_checkpoints(output_dir: str = "clip"):
    """
    List available checkpoints for resuming training.
    """
    clip_dir = Path(__file__).parent / output_dir
    if not clip_dir.exists():
        print(f"❌ Output directory {clip_dir} does not exist")
        return
    
    checkpoints = []
    
    # Find checkpoint directories
    for item in clip_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
                checkpoints.append((step, item))
            except ValueError:
                continue
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {clip_dir}")
        return
    
    print(f"📁 Available checkpoints in {clip_dir}:")
    for step, checkpoint_path in checkpoints:
        print(f"   Step {step:>6}: {checkpoint_path}")
    
    # Show best checkpoint if it exists
    best_checkpoint = clip_dir / "best_checkpoint"
    if best_checkpoint.exists():
        accuracy_file = best_checkpoint / "best_accuracy.txt"
        accuracy_info = ""
        if accuracy_file.exists():
            try:
                with open(accuracy_file, "r") as f:
                    lines = f.readlines()
                    accuracy = lines[0].strip().split(":")[1].strip()
                    accuracy_info = f" (accuracy: {accuracy})"
            except:
                pass
        print(f"   🏆 Best: {best_checkpoint}{accuracy_info}")
    
    print(f"\n💡 To resume training from a checkpoint:")
    print(f"   python -m homework.clip train --resume_from_checkpoint {checkpoints[-1][1]}")
    print(f"💡 To resume from the best checkpoint:")
    if best_checkpoint.exists():
        print(f"   python -m homework.clip train --resume_from_checkpoint {best_checkpoint}")


def main():
    from fire import Fire

    Fire({"train": train, "test": test, "list_checkpoints": list_checkpoints})


if __name__ == "__main__":
    main()
