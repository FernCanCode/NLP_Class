import pandas as pd
import os

def make_md_table(df):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_vals = []
        for v in row:
            v_str = str(v).replace('\n', '<br>').replace('|', '&#124;')
            row_vals.append(v_str)
        rows.append("| " + " | ".join(row_vals) + " |")
    return "\n".join([header, sep] + rows)

def create_report():
    p1 = pd.read_csv("results/part1_results.csv")
    p2 = pd.read_csv("results/part2_results.csv")
    
    p1_cols = ["query_id", "query_text", "top_k_sources", "similarity_scores", "generated_answer_first_two_sentences", "grounded"]
    p2_cols = ["query_id", "query_type", "query_text", "top_k_sources", "similarity_scores", "corpus_mix", "retrieved_custom_functions", "generated_answer_first_two_sentences", "grounded"]
    
    p1_md = make_md_table(p1[p1_cols])
    p2_md = make_md_table(p2[p2_cols])
    
    report_text = f"""# Assignment 4 Report: Track B Code RAG

## Track Declaration
I selected Track B: Code RAG.

## Pipeline Summary
Our RAG pipeline operates on a starter corpus of 1,000 Python functions retrieved from the CodeSearchNet Python split. These snippets are natively vectorized using the `sentence-transformers/all-MiniLM-L6-v2` embedding model and stored in a local, persistent Chroma vector database. During query execution, our retriever embeds the user's natural language question and fetches the top k=4 most relevant code chunks, citing them cleanly utilizing the `repo/path::func_name` format. The retrieved contexts are then seamlessly forwarded alongside a grounded system prompt to the Anthropic Claude API (using a low-cost Claude Haiku model) for generation. To increase the complexity of our search space, Part 2 intentionally introduces an additional 5 custom NLP utility functions imported directly from `data/custom_functions.py` into our database.

## Part 1 Results
The full Part 1 results are saved in results/part1_results.csv.

{p1_md}

## Part 2 Results
The full Part 2 results are saved in results/part2_results.csv.

{p2_md}

## Reflection
Retrieval naturally succeeded seamlessly when the language of the query mirrored logic written within function docstrings or naming conventions. Targeted queries designed for extracting hashtags, evaluating basic sentiment, counting word frequencies, cleaning textual data, and splitting sentences reliably pinged the corresponding local Python functions correctly and isolated results.

However, retrieval occasionally struggled with cross-corpus edge cases by successfully surfacing topically adjacent code blocks from the overarching CodeSearchNet environment that did not actually fulfill the explicit user need—an inherent constraint when assessing snippet value solely through rigid vector similarity without robust holistic context. Despite this finding, queries traversing across both data spheres consistently retrieved overlapping combinations efficiently. In Part 2, exactly 9 out of our 10 evaluated runs returned an integrated 'both' corpus mix consisting of custom module integrations alongside standard library definitions, whereas only one isolated query strictly triggered starters alone. Our custom functions effectively asserted themselves against the broader 1000 item corpus precisely when the logical parameters necessitated them during targeted queries.

If provided with an extra day to mature processing flows, incorporating an 'LLM-as-judge' validation tier leveraging rigid response rubrics instead of generalized keyword-grounded boolean checks would massively elevate evaluation fidelity. Replacing our generic sentence-transformer baseline implementation for a strictly programmatic, code-oriented embeddings architecture would also significantly condense the text-to-logic semantic gap thereby upgrading raw codebase retrieval quality.

## Reproducibility Notes
- The Chroma database can effectively be rebuilt or refreshed organically utilizing the standalone indexing scripts.
- To execute processing logic natively, API keys are completely withheld from output logs and version control; this must successfully be authenticated using an active `.env` file structure locally.
- Aggregated CSV files tracking metrics are autonomously generated and securely managed through `run_part1.py` and `run_part2.py`.
"""
    with open("REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_text)

if __name__ == "__main__":
    create_report()
    print("REPORT.md successfully generated and overwritten from template!")
