# Sequential Fine-Tuning: From Generalist to JSON Specialist
**Master of Science in AI (MSAI) Project**

This repository contains the codebase and methodology for a two-stage instruction tuning pipeline on the `microsoft/Phi-3.5-mini-instruct` (3.8B) model using LoRA. The project demonstrates how to specialize a generalist LLM into a highly accurate JSON tool-calling specialist without suffering from catastrophic forgetting.

## Project Overview

*   **Model**: `microsoft/Phi-3.5-mini-instruct`
*   **Infrastructure**: NVIDIA H100 SXM (80GB VRAM). The project relies on BFloat16 precision instead of 4-bit QLoRA to bypass library compatibility issues and maximize hardware throughput.
*   **Sequential Pipeline**:
    1.  **Stage 1**: Fine-tuning on 5,000 conversational instruction-following samples (Alpaca).
    2.  **Stage 2 (The Power Run)**: Fine-tuning on 100 generated JSON Tool-Call samples using high-intensity parameters (15 epochs, 1e-4 learning rate) to combat task interference and enforce strict JSON schema adherence.
*   **Evaluation Methodology**:
    *   **Robust Parsing**: Implements a Regex-based extraction parser to correctly evaluate JSON returned alongside reasoning or Markdown formatting.
    *   **Strong-Model Judge**: Leverages `Llama-3.1-70B-Instruct` via the OpenRouter API to perform rigorous "Forgetting Analysis."

## Results Summary

1.  **JSON Validity Matrix**:
    *   **Base Model (C0)**: 65.9%
    *   **Post-Alpaca (C1)**: 73.4%
    *   **Post-JSON (C2)**: **98.9%** 
2.  **Forgetting Analysis**:
    *   The model demonstrated strong retention of conversational instruction following after specialization, capturing a **46% tie rate** against its previous checkpoint on general tasks via the LLM-as-a-judge protocol.

---

## Setup & Installation

### Requirements
*   Python 3.10+
*   HPC Environment (or sufficient VRAM for BFloat16 fine-tuning of a 3.8B model)

### Dependencies
Create a virtual environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers peft accelerate datasets evaluate rouge_score bert_score requests tqdm matplotlib python-dotenv
```

### API Keys
Because the pipeline uses Imitation Learning for dataset generation and a strong-model Judge for evaluations, you must supply an OpenRouter API Key.

1.  Create a file named `.env` in the root directory.
2.  Add your key:
    ```env
    OPENROUTER_API_KEY=your_key_here
    ```

---

## Repository Structure

The codebase is highly modular, split by pipeline phase:

*   **`config.py`**: Central source-of-truth for hyperparameters, paths, the robust JSON parser, and all editable text-generation prompt templates.
*   **`data_prep/`**:
    *   `prepare_data.py`: Downloads and parses the Alpaca dataset.
    *   `fix_json.py` & `topup_json_repair.py`: Generates the structured tool-calling dataset via imitation learning from Llama-3.1-70B.
*   **`training/`**: 
    *   `train_stage1.py`: Alpaca training.
    *   `train_stage2.py`: The specialized JSON "Power Run" training.
*   **`evaluation/`**: 
    *   `evaluate_checkpoints.py`: Unified script calculating JSON validity and ROUGE scores.
    *   `judge_eval.py`: LLM-as-a-judge forgetting analysis script.
    *   `generate_plots.py`: Creates loss curves and evaluation graphs.
*   **`scripts/`**: Ready-to-go `.slurm` HPC batch scripts using UTSA Arc module parameters.
*   **`export_project.py`**: Artifact packaging utility.

---

## Reproduction Steps

To faithfully reproduce the pipeline results, execute scripts from the **project root directory** (to assure python module paths resolve automatically). 

*If running these interactively, ensure your `.venv` is activated and run with `PYTHONPATH=.`.*

1.  **Prepare the Data**:
    ```bash
    python data_prep/prepare_data.py
    python data_prep/fix_json.py
    python data_prep/json_validation.py
    ```
2.  **Stage 1 - Generalist Fine-Tuning**:
    Run `training/train_stage1.py` manually, or submit the provided Slurm job:
    ```bash
    sbatch scripts/run_stage1.slurm
    ```
3.  **Stage 2 - Specialist Fine-Tuning**:
    *(Expects Stage 1 adapter to be located in `./adapters`)*
    ```bash
    sbatch scripts/run_stage2.slurm
    ```
4.  **Evaluate Unified Metrics**:
    Generate JSON validity tests for the baseline and generated models:
    ```bash
    python evaluation/evaluate_checkpoints.py
    ```
5.  **Evaluate Forgetting (LLM Judge)**:
    Compare conversational and reasoning retention using OpenRouter:
    ```bash
    python evaluation/judge_eval.py
    ```
6.  **Visualize Results**:
    Generates `.png` graphs representing loss curves and the judge win-ration matrix.
    ```bash
    python evaluation/generate_plots.py
    ```
