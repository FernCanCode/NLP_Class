# export_project.py
import zipfile, os

files = ["loss_curve.png", "judge_win_rates.png", "final_judge_results.json", "config.py"]
with zipfile.ZipFile("msai_project_final.zip", 'w') as zipf:
    for f in files:
        if os.path.exists(f): zipf.write(f)
    for d in ["data_prep", "training", "evaluation", "scripts", "adapters"]:
        for root, dirs, filenames in os.walk(f"./{d}"):
            for filename in filenames:
                if "__pycache__" not in root:
                    zipf.write(os.path.join(root, filename))
print("Final submission zip created.")