from datasets import Dataset, load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from helper import display_dataset, load_model_and_tokenizer

USE_GPU = True

model_name = "HuggingFaceTB/SmolLM2-135M"
# model_name = "Qwen/Qwen3-0.6B-Base"
model, tokenizer = load_model_and_tokenizer(model_name, use_gpu=USE_GPU)

train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]
display_dataset(train_dataset)

sft_config = SFTConfig(
    learning_rate=8e-5,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    logging_steps=10,
)

sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
sft_trainer.train()

# save the model
sft_trainer.save_model("models/SmolLM2-135M-SFT")
