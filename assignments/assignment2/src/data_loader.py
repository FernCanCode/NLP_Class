import os
import json
from datasets import load_dataset

# Configuration for pathing
DATA_DIR = "data"
DATA_FILENAME = os.path.join(DATA_DIR, "strategy_qa_sample.json")

def load_and_sample_data(n_samples=150):
    """
    Checks for local data in the data/ folder; if missing, downloads 
    StrategyQA, samples it, and saves it locally.
    """
    # 1. Create the data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created directory: {DATA_DIR}")

    # 2. Check if the sample already exists locally
    if os.path.exists(DATA_FILENAME):
        print(f"✅ Found local data at {DATA_FILENAME}. Loading...")
        with open(DATA_FILENAME, "r") as f:
            return json.load(f)

    # 3. Download from Hugging Face if not found
    print("🚀 Local data not found. Downloading StrategyQA...")
    try:
        dataset = load_dataset("tasksource/strategy-qa", split="train")
        
        # 4. Sample 100-200 questions [cite: 24, 64]
        sampled_data = dataset.shuffle(seed=42).select(range(n_samples))
        
        data_list = []
        for item in sampled_data:
            data_list.append({
                "qid": item.get("qid"),
                "question": item.get("question"),
                "answer": item.get("answer") # Ground truth [cite: 40]
            })
            
        # 5. Save to the data/ folder
        with open(DATA_FILENAME, "w") as f:
            json.dump(data_list, f, indent=4)
            
        print(f"💾 Successfully saved {n_samples} samples to {DATA_FILENAME}.")
        return data_list

    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

if __name__ == "__main__":
    data = load_and_sample_data(150)
    if data:
        print(f"Successfully loaded {len(data)} questions.")