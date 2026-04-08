import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate import load
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_ID, STAGE1_ADAPTER

# --- CONFIGURATION ---
alp_test_file = "alpaca_eval.json"
jsn_test_file = "json_eval.json"

print("--- 📥 Loading Checkpoint 1 (Base Model + Alpaca Adapter) ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load base model in BFloat16
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load the LoRA adapter we just trained
model = PeftModel.from_pretrained(base_model, STAGE1_ADAPTER)
model.eval()

# --- METRIC SETUP ---
rouge = load("rouge")
bertscore = load("bertscore")

def generate_response(prompt):
    formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()

# --- 📊 PART A: ALPACA EVALUATION (Row 2, Cols 1-2) ---
print("\n--- 📝 Running Alpaca Evaluation ---")
with open(alp_test_file, 'r') as f: alp_data = json.load(f)

predictions = []
references = []
# We'll save these for the Judge later
checkpoint1_outputs = []

for sample in tqdm(alp_data[:100], desc="Alpaca Eval"):
    full_prompt = f"{sample['instruction']}\n{sample.get('input', '')}"
    pred = generate_response(full_prompt)
    predictions.append(pred)
    references.append(sample['output'])
    checkpoint1_outputs.append({"instruction": full_prompt, "response": pred})

rouge_results = rouge.compute(predictions=predictions, references=references)
bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")

# Save outputs for Judge comparison later
with open("outputs_checkpoint1_alpaca.json", "w") as f:
    json.dump(checkpoint1_outputs, f, indent=4)

# --- 📊 PART B: JSON EVALUATION (Row 2, Cols 3-5) ---
print("\n--- 🛠️ Running JSON Evaluation ---")
with open(jsn_test_file, 'r') as f: jsn_data = json.load(f)

valid_json_count = 0
for sample in tqdm(jsn_data[:100], desc="JSON Validity"):
    pred = generate_response(sample['instruction'])
    try:
        json.loads(pred)
        valid_json_count += 1
    except:
        continue

# --- 🏁 FINAL OUTPUT FOR REPORT ---
print("\n" + "="*50)
print("🏆 CHECKPOINT 1 (AFTER STAGE 1) RESULTS")
print("="*50)
print(f"Alpaca ROUGE-L:  {rouge_results['rougeL']:.4f}")
print(f"Alpaca BERTScore: {sum(bert_results['f1'])/len(bert_results['f1']):.4f}")
print(f"JSON Validity:   {(valid_json_count/100)*100:.2f}%")
print("="*50)