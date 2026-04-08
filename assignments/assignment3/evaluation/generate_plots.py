# generate_plots.py
import json
import matplotlib.pyplot as plt

def plot_loss():
    with open("stage1_logs.json", "r") as f: s1 = json.load(f)
    with open("stage2_logs.json", "r") as f: s2 = json.load(f)
    
    l1 = [x['loss'] for x in s1 if 'loss' in x]
    l2 = [x['loss'] for x in s2 if 'loss' in x]

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(l1)), l1, label='Stage 1: Alpaca')
    plt.plot(range(len(l1), len(l1)+len(l2)), l2, label='Stage 2: JSON Specialist', color='red')
    plt.title("Sequential Training Loss")
    plt.legend()
    plt.savefig("loss_curve.png")

def plot_wins():
    labels = ['C1 (Alpaca)', 'C2 (JSON)', 'Ties']
    values = [28, 26, 46]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['blue', 'red', 'gray'])
    plt.title("Judge Win Rates")
    plt.savefig("judge_win_rates.png")

plot_loss()
plot_wins()