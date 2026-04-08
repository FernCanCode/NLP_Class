# train_stage1.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_ID, LORA_CONFIG, STAGE1_ADAPTER

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
model = get_peft_model(model, LoraConfig(**LORA_CONFIG))

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

def tokenize(prompt):
    full_text = f"<|user|>\n{prompt['instruction']}\n{prompt['input']}<|end|>\n<|assistant|>\n{prompt['output']}<|end|>"
    return tokenizer(full_text, truncation=True, max_length=512, padding=False)

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

args = TrainingArguments(
    output_dir="./outputs/stage1",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
    save_strategy="no"
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset, data_collator=DataCollatorForSeq2Seq(tokenizer))
trainer.train()
model.save_pretrained(STAGE1_ADAPTER)