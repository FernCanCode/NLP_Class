# judge_eval.py
import json, random, requests
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import JUDGE_PROMPT_TEMPLATE
API_KEY = "YOUR_OPENROUTER_KEY"
MODEL = "meta-llama/llama-3.1-70b-instruct"

with open("outputs_checkpoint1_alpaca.json", "r") as f: c1_data = json.load(f)
with open("outputs_checkpoint2_json.json", "r") as f: c2_data = json.load(f)

results = []
stats = {"C1_Wins": 0, "C2_Wins": 0, "Ties": 0}

for i in tqdm(range(50)):
    prompt_text = c1_data[i]["instruction"]
    variants = [{"id": "C1", "text": c1_data[i]["response"]}, {"id": "C2", "text": c2_data[i]["response"]}]
    random.shuffle(variants)
    
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt_text=prompt_text, response_a=variants[0]['text'], response_b=variants[1]['text'])
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": [{"role": "user", "content": judge_prompt}]}
    )
    
    res = response.json()['choices'][0]['message']['content']
    # Minimal logic to extract winner from string
    winner = "Tie"
    if "'winner': 'A'" in res or '"winner": "A"' in res: winner = variants[0]["id"]
    elif "'winner': 'B'" in res or '"winner": "B"' in res: winner = variants[1]["id"]
    
    stats[f"{winner if winner == 'Tie' else winner+'_Wins'}"] += 1
    results.append({"winner": winner})

print(stats)