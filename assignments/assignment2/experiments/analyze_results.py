import json
import matplotlib.pyplot as plt
import os

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return []
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
    # Load all three result files
    debate_results = load_json("data/results_log.json")
    zero_shot_results = load_json("data/zero_shot_results.json")
    few_shot_results = load_json("data/few_shot_results.json")

    # We need to map questions to their accuracies. 
    # Because baseline scripts evaluated 30 questions, let's use the zero-shot questions as the anchor.
    evaluated_questions = [item['question'] for item in zero_shot_results]

    stats = {
        "Debate": {"correct": 0, "total": 0},
        "Zero-Shot Direct QA": {"correct": 0, "total": 0},
        "Zero-Shot Self-Consistency": {"correct": 0, "total": 0},
        "Few-Shot Direct QA": {"correct": 0, "total": 0},
        "Few-Shot Self-Consistency": {"correct": 0, "total": 0},
    }

    # Process Zero-Shot
    for item in zero_shot_results:
        q = item["question"]
        gt = normalize_bool_to_yes_no(item["ground_truth"])
        
        # Direct QA
        dqa_verdict = item["direct_qa"].get("majority_verdict")
        if dqa_verdict == gt:
            stats["Zero-Shot Direct QA"]["correct"] += 1
        stats["Zero-Shot Direct QA"]["total"] += 1
        
        # Self-Consistency
        sc_verdict = item["self_consistency"].get("majority_verdict")
        if sc_verdict == gt:
            stats["Zero-Shot Self-Consistency"]["correct"] += 1
        stats["Zero-Shot Self-Consistency"]["total"] += 1

    # Process Few-Shot
    few_shot_dict = {item["question"]: item for item in few_shot_results}
    for q in evaluated_questions:
        if q in few_shot_dict:
            item = few_shot_dict[q]
            gt = normalize_bool_to_yes_no(item["ground_truth"])
            
            dqa_verdict = item["direct_qa"].get("majority_verdict")
            if dqa_verdict == gt:
                stats["Few-Shot Direct QA"]["correct"] += 1
            stats["Few-Shot Direct QA"]["total"] += 1
            
            sc_verdict = item["self_consistency"].get("majority_verdict")
            if sc_verdict == gt:
                stats["Few-Shot Self-Consistency"]["correct"] += 1
            stats["Few-Shot Self-Consistency"]["total"] += 1

    # Process Debate
    # The debate output verdict is in `extracted_verdict`
    debate_dict = {item["question"]: item for item in debate_results}
    for q in evaluated_questions:
        if q in debate_dict:
            item = debate_dict[q]
            gt = normalize_bool_to_yes_no(item["ground_truth"])
            
            # Depending on if it was successfully extracted
            debate_verdict = item.get("extracted_verdict")
            if debate_verdict == gt:
                stats["Debate"]["correct"] += 1
            stats["Debate"]["total"] += 1

    # Output Raw Numbers
    print("=== RAW NUMERICAL STATISTICS ===")
    accuracies = {}
    for method, counts in stats.items():
        if counts["total"] > 0:
            accuracy = (counts["correct"] / counts["total"]) * 100
            accuracies[method] = accuracy
            print(f"{method}: {counts['correct']}/{counts['total']} correct ({accuracy:.1f}%)")
        else:
            accuracies[method] = 0.0
            print(f"{method}: N/A (0 total)")
    print("================================")

    if not accuracies:
        print("No data available to plot.")
        return

    # Create Chart
    # Ensure experiments folder exists
    os.makedirs("experiments", exist_ok=True)
    
    methods = list(accuracies.keys())
    values = list(accuracies.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974'])
    
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Performance Comparison: Debate vs Baselines', fontsize=14, pad=20)
    plt.ylim(0, 100)
    
    # Add annotations on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Rotate x labels for better readability
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()

    # Save chart
    chart_path = "experiments/accuracy_comparison.png"
    plt.savefig(chart_path, dpi=300)
    print(f"\nChart successfully saved to {chart_path}!")

if __name__ == "__main__":
    main()
