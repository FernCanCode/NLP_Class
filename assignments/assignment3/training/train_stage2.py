# train_stage2.py
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import PeftModel
from datasets import load_dataset
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_ID, STAGE1_ADAPTER, STAGE2_ADAPTER

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base_model, STAGE1_ADAPTER, is_trainable=True)

dataset = load_dataset("json", data_files="json_train.json", split="train")

def tokenize(example):
    output = example['output'] if isinstance(example['output'], str) else json.dumps(example['output'])
    full_prompt = f"<|user|>\n{example['instruction']}<|end|>\n<|assistant|>\n{output}<|end|>"
    res = tokenizer(full_prompt, truncation=True, max_length=1024, padding=False)
    res["labels"] = res["input_ids"].copy()
    return res

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

args = TrainingArguments(
    output_dir="./outputs/stage2",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4, # Power Run LR
    num_train_epochs=15, # Power Run Epochs
    bf16=True,
    logging_steps=5,
    save_strategy="no"
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset, data_collator=DataCollatorForSeq2Seq(tokenizer))
trainer.train()
model.save_pretrained(STAGE2_ADAPTER)