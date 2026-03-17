import json
import matplotlib.pyplot as plt
import os

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)

def normalize_bool_to_yes_no(val):
    if val is True: return "Yes"
    if val is False: return "No"
    if isinstance(val, str):
        if val.lower() == 'true': return "Yes"
        if val.lower() == 'false': return "No"
    return str(val).capitalize()

def main():
    ablation_results = load_json("data/ablation_results.json")
    
    if not ablation_results:
        print("No ablation results found or file empty.")
        return
        
    accuracies = []
    round_labels = []
    
    # Needs to be sorted numerically, keys might be strings depending on JSON serialize
    keys_sorted = sorted([int(k) for k in ablation_results.keys()])
    
    print("=== ABLATION SCALING STATS ===")
    for k_int in keys_sorted:
        k_str = str(k_int)
        runs = ablation_results.get(k_str)
        if not runs:
            continue
            
        correct = 0
        total = 0
        
        for item in runs:
            gt = normalize_bool_to_yes_no(item["ground_truth"])
            verdict = item.get("extracted_verdict")
            if verdict == gt:
                correct += 1
            total += 1
            
        if total > 0:
            acc = (correct / total) * 100
        else:
            acc = 0.0
            
        accuracies.append(acc)
        round_labels.append(k_str)
        print(f"Max Rounds {k_str}: {acc:.1f}% ({correct}/{total})")
        
    os.makedirs("experiments", exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    # Plot line chart
    plt.plot(round_labels, accuracies, marker='o', linestyle='-', color='#C44E52', linewidth=2, markersize=8)
    
    plt.xlabel('Max Debate Rounds (Length Penalty)', fontsize=12)
    plt.ylabel('Aggregate Accuracy (%)', fontsize=12)
    plt.title('Ablation Study: Debate Round Scaling', fontsize=14, pad=20)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    for i, acc in enumerate(accuracies):
        plt.annotate(f'{acc:.1f}%', (round_labels[i], acc), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
        
    plt.tight_layout()
    chart_path = "experiments/ablation_scaling.png"
    plt.savefig(chart_path, dpi=300)
    print(f"\nChart successfully saved to {chart_path}!")

if __name__ == "__main__":
    main()
