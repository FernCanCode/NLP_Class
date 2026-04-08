import json
from openai import OpenAI
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TEACHER_PROMPT_TOPUP
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="YOUR_OPENROUTER_KEY")

# Focus on the 'JSON Repair' task to balance the set
task = "JSON repair or formatting correction"
additional_samples = []

print("--- 🛠️ Topping up JSON Eval Set to 100 samples ---")
while len(additional_samples) < 6:
    prompt = TEACHER_PROMPT_TOPUP.format(task=task)
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        sample = json.loads(response.choices[0].message.content)
        if isinstance(sample.get('output'), (dict, list)):
            sample['output'] = json.dumps(sample['output'])
        json.loads(sample['output'])
        additional_samples.append(sample)
        print(f"Added sample {len(additional_samples)}/6")
    except:
        continue

# Append to our existing file
with open("json_eval.json", "r") as f:
    eval_set = json.load(f)

eval_set.extend(additional_samples)

with open("json_eval.json", "w") as f:
    json.dump(eval_set, f, indent=4)

print(f"✅ Final JSON Eval Count: {len(eval_set)}")