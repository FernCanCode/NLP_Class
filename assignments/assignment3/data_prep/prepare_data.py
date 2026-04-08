import json
import os
import time
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TEACHER_PROMPT_TEMPLATE

load_dotenv()

# --- CONFIGURATION ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_KEY_HERE")
TEACHER_MODEL = "meta-llama/llama-3.1-70b-instruct" 
ALPAC_DATASET_NAME = "yahma/alpaca-cleaned" 

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def prepare_alpaca():
    print("--- Step 1: Preparing Alpaca Data ---")
    # Download and normalize into (instruction, input, output)
    ds = load_dataset(ALPAC_DATASET_NAME, split="train")
    
    # We only need a subset for this speedrun, but we must set aside 100 for eval
    # We'll take 5000 for training and 100 for a held-out evaluation set
    data = [{"instruction": x["instruction"], "input": x["input"], "output": x["output"]} for x in ds]
    
    train_split = data[:5000]
    eval_split = data[-100:]
    
    with open("alpaca_train.json", "w") as f:
        json.dump(train_split, f, indent=4)
    with open("alpaca_eval.json", "w") as f:
        json.dump(eval_split, f, indent=4)
    print(f"Saved {len(train_split)} training and {len(eval_split)} eval samples.")

def generate_json_tasks():
    print("--- Step 2: Generating Teacher JSON Data ---")
    # The 5 required task types
    task_types = [
        "JSON extraction from unstructured text",
        "Schema-constrained generation",
        "Exact-label classification with JSON output",
        "JSON repair or formatting correction",
        "Tool-call argument generation"
    ]
    
    json_instruct_data = []
    
    # We need 100 total prompts (20 per task type)
    for task in task_types:
        print(f"Generating samples for: {task}")
        for i in range(20):
            prompt_content = TEACHER_PROMPT_TEMPLATE.format(task=task)
            
            try:
                response = client.chat.completions.create(
                    model=TEACHER_MODEL,
                    messages=[{"role": "user", "content": prompt_content}],
                    response_format={"type": "json_object"}
                )
                
                # Validate JSON correctness
                sample = json.loads(response.choices[0].message.content)
                
                # Ensure the 'output' within the sample is also valid JSON text as required
                # some tasks might require the output field itself to be a stringified JSON
                json_instruct_data.append(sample)
                time.sleep(0.5) # Avoid rate limits
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue

    # Split into 80 train / 20 eval for the JSON set
    with open("json_train.json", "w") as f:
        json.dump(json_instruct_data[:80], f, indent=4)
    with open("json_eval.json", "w") as f:
        json.dump(json_instruct_data[80:], f, indent=4)
    print("JSON Data Generation Complete.")

if __name__ == "__main__":
    prepare_alpaca()
    generate_json_tasks()