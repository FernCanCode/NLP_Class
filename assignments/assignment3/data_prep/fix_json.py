import json
import random
from openai import OpenAI
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TEACHER_PROMPT_FIX
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_KEY")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

task_types = [
    "JSON extraction from unstructured text",
    "Schema-constrained generation",
    "Exact-label classification with JSON output",
    "JSON repair or formatting correction",
    "Tool-call argument generation"
]

all_samples = []

print("--- 🚀 Generating Balanced JSON Dataset (200 samples) ---")
for task in task_types:
    print(f"Generating 40 samples for: {task}...")
    for i in range(40):
        # Improved prompt to ensure the "output" is stringified JSON within the final object
        prompt = TEACHER_PROMPT_FIX.format(task=task)
        
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            sample = json.loads(response.choices[0].message.content)
            # Ensure 'output' is a string. If the API sent a dict, stringify it.
            if isinstance(sample.get('output'), (dict, list)):
                sample['output'] = json.dumps(sample['output'])
            
            # Final validation: can we parse the 'output'?
            json.loads(sample['output'])
            all_samples.append(sample)
            time.sleep(0.2)
        except Exception as e:
            print(f"Error: {e}")

# Shuffle to ensure both train/eval see all task types
random.shuffle(all_samples)

# Split 100/100 to meet rubric requirements 
json_train = all_samples[:100]
json_eval = all_samples[100:]

with open("json_train.json", "w") as f: json.dump(json_train, f, indent=4)
with open("json_eval.json", "w") as f: json.dump(json_eval, f, indent=4)

print(f"✅ Done! Saved {len(json_train)} train and {len(json_eval)} eval samples.")