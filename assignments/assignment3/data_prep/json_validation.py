import json

def validate_data(alpaca_path, json_path):
    print("--- 🔍 Final Validation ---")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    required = ["extract", "schema", "classif", "repair", "tool"]
    found_tasks = {r: 0 for r in required}
    
    for i, s in enumerate(data):
        # 1. Check Schema 
        if not all(k in s for k in ['instruction', 'input', 'output']):
            print(f"❌ Sample {i} missing keys.")
            return

        # 2. Check JSON validity in output
        try:
            # If it's already a dict, stringify it for the model; if string, verify it
            if isinstance(s['output'], dict):
                s['output'] = json.dumps(s['output'])
            json.loads(s['output'])
        except:
            print(f"❌ Sample {i} has invalid JSON output.")
            return

        # 3. Track task types
        instr = s['instruction'].lower()
        for r in required:
            if r in instr: found_tasks[r] += 1

    print(f"✅ Total Samples: {len(data)}")
    print(f"📊 Task Distribution: {found_tasks}")
    print("🔥 Ready to upload to RunPod.")

validate_data("alpaca_train.json", "json_eval.json")