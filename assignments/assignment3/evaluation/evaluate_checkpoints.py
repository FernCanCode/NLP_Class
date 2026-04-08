# evaluate_checkpoints.py
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evaluate import load
from tqdm import tqdm
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_ID, robust_json_parser

def run_eval(adapter_path=None, name="Base"):
    print(f"--- Testing {name} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # JSON Eval
    with open("json_eval.json", "r") as f: data = json.load(f)
    valid = 0
    for s in tqdm(data):
        prompt = f"<|user|>\n{s['instruction']}<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256)
        resp = tokenizer.decode(out[0], skip_special_tokens=False).split("<|assistant|>")[-1].replace("<|end|>", "").strip()
        if robust_json_parser(resp): valid += 1
    
    print(f"{name} JSON Validity: {(valid/len(data))*100:.2f}%")

# Example usage:
# run_eval(None, "C0")
# run_eval("./adapters/checkpoint1_alpaca", "C1")
# run_eval("./adapters/checkpoint2_json", "C2")