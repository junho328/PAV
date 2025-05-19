from openai import OpenAI
import os
import argparse
from pathlib import Path
import json
from graphviz import Source  # pip install graphviz

# YOU SHOULD MAKE ACTION SEQUENCES BEFORE RUNNING THIS CODE

GPT_PROMPT = """You are an expert action-graph builder.

TASK
-----
Given a collection of macro-action sequences (each sequence is an ordered list of actions taken on a mobile app), build a **directed action-transition graph** that captures which action can be followed by which.

Rules
1. **Focus only on the action type**, not on object names or specific criteria.
2. Merge actions that share the same type into a single node.
3. For every adjacent pair of actions (A, B) within a sequence, draw a directed edge A → B.
4. Self-loops are not needed. Duplicate edges should appear only once.
5. **Do not invent or infer any action types.** Only use action types that appear in the input macro-action sequences.
6. Output **only** the graph in valid **DOT language**, wrapped in 
digraph {{
    ...
}} 

No explanatory text before or after the graph.

Input Sequences
---------------
{sequences_block}

(End of prompt)"""

def main(args):
    
    client = OpenAI(api_key=args.api_key)

    with open(args.action_sequences, 'r') as f:
        action_sequences = json.load(f)
        
    seq_lines = ["[" + ", ".join(seq) + "]" for seq in action_sequences]
    sequences_block = "\n".join(seq_lines)

    response = client.chat.completions.create(
    model="gpt-4o",   
    messages=[{"role": "user", "content": GPT_PROMPT.format(sequences_block=sequences_block)}],
    temperature=0          
    )

    dot_graph = response.choices[0].message.content.strip()
    print("=== DOT Graph ===")
    print(dot_graph)
    
    with open("action_graph.dot", "w", encoding="utf-8") as f:
        f.write(dot_graph)
    
    visualize_digraph = dot_graph.split("dot")[1].split("```")[0].strip()

    src = Source(visualize_digraph, format="png")
    
    src.render("action_graph", view=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate action graph from action_sequences.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--action_sequences", type=str, required=True, help="Path to the action sequences JSON file")
    parser.add_argument("--app_name", type=str, default="google")
    
    args = parser.parse_args()
    
    main(args)
