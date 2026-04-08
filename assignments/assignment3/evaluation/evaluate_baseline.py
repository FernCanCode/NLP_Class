import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from evaluate import load
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_ID, robust_json_parser

def run_baseline_eval():
    print("--- 📊 Running Checkpoint 0 (Baseline) Evaluation ---")
    
    # 1. Load Model and Tokenizer 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config, 
        device_map="auto"
    )

    # 2. Load Evaluation Sets
    with open("json_eval.json", 'r') as f: json_test = json.load(f)
    
    # 3. JSON Validity Evaluation
    valid_count = 0
    print("Evaluating JSON Validity...")
    for sample in json_test:
        prompt = f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:\n")[-1]
        
        try:
            json.loads(response.strip())
            valid_count += 1
        except:
            continue

    print(f"✅ Baseline JSON Validity: {(valid_count/len(json_test))*100:.2f}%")
    
    # Note: ROUGE/BERTScore for Alpaca will be handled in the final consolidated script 
    # to save time, but the JSON validity is the key 'Check 0' metric needed now.

if __name__ == "__main__":
    run_baseline_eval()