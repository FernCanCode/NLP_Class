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
This project builds a RAG pipeline using a starter corpus of 1,000 Python functions from the CodeSearchNet dataset. The code snippets are embedded using the `sentence-transformers/all-MiniLM-L6-v2` model and saved in a local Chroma vector database. When a query is run, the retriever embeds the question and pulls the top k=4 related code chunks, citing them using the `repo/path::func_name` format. The retrieved chunks are then passed into a grounded prompt for the Anthropic Claude API (specifically the Claude Haiku model) to generate an answer. For Part 2, I also added 5 custom NLP utility functions from `data/custom_functions.py` to the database.

## Part 1 Results
The full Part 1 results are saved in results/part1_results.csv.

{p1_md}

## Part 2 Results
The full Part 2 results are saved in results/part2_results.csv.

{p2_md}

## Reflection
Retrieval worked best when the wording of the query closely matched the function names or docstrings. For the targeted queries, it reliably found the right custom functions for things like extracting hashtags, getting sentiment, counting word frequencies, cleaning text, and splitting sentences.

However, the retrieval sometimes struggled with broader questions. It would return chunks that seemed related based on keywords but didn't actually answer the prompt. This shows the limitations of simple vector similarity for code without deeper context. For the cross-corpus queries, the system mostly retrieved a mix of both datasets. In the Part 2 run, 9 out of 10 queries had a corpus mix of "both", while 1 only retrieved "starter" functions. The custom functions showed up when they were supposed to.

If I had another day to work on this, I would try to add a better evaluation method. Instead of my simple keyword check for grounding, using an LLM-as-judge setup with a rubric, or doing a manual review, would give better insights. Another thing to try would be using a code-specialized embedding model instead of `all-MiniLM-L6-v2`, since that might understand the actual code structure much better.

## Reproducibility Notes
- The Chroma database can be deleted and rebuilt from scratch using the provided scripts.
- My Anthropic API key is not included in the repo. You will need to put your own key in a `.env` file to run the generator scripts.
- The results CSVs are generated automatically by running `run_part1.py` and `run_part2.py`.

## AI Use Acknowledgment
AI was actively utilized throughout the development of this project. Specifically, AI assistance was used for all of the code generation within the Python pipeline, the generation of the `README.md` file, and for formatting and styling this `REPORT.md` deliverable.
"""
    with open("REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_text)

if __name__ == "__main__":
    create_report()
    print("REPORT.md successfully generated and overwritten from template!")
