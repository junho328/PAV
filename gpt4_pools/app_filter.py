import os
import json
import numpy as np
import base64
import io

from openai import OpenAI
from sentence_transformers import SentenceTransformer

import argparse

# --- Utility Functions ---

def cosine_similarity(vec_a, vec_b):
    """Calculates the cosine similarity between two vectors."""
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image file to a Base64 string for API usage."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def preprocess_json(json_data):
    
    list_data = []
    
    for key1, value1 in json_data.items():
        subset = key1
        for key2, value2 in value1.items():
            dict_data = {}
            dict_data['id'] = subset + '_' + key2
            dict_data['instruction'] = value2.get('instruction', '')
            dict_data['macro_actions'] = value2.get('macro_actions', [])
            list_data.append(dict_data)
            
    return list_data

# --- Stage 1: Semantic Similarity Filtering using Hugging Face Jina ---

def filter_by_semantic_similarity(
    task_pool: list, 
    app_description: str, 
    threshold: float,
    model: SentenceTransformer
) -> list:
    """
    Evaluates semantic relevance using a Hugging Face Jina embedding model.
    """
    print(f"\n[Stage 1] Starting semantic similarity filtering using Hugging Face model: '{model.model_name_or_path}'...")
    initial_candidates = []
    
    try:
        app_vector = model.encode(app_description)
        task_texts = [task['instruction'] for task in task_pool]
        task_vectors = model.encode(task_texts)

        print(f"Similarity Threshold: {threshold}")
        for i, task in enumerate(task_pool):
            similarity = cosine_similarity(app_vector, task_vectors[i])
            print(f"  - Task ID {task['id']}: '{task['instruction']}...' -> Similarity: {similarity:.4f}")
            
            if similarity >= threshold:
                initial_candidates.append(task)
                print(f"    -> PASSED (Similarity >= {threshold})")
    except Exception as e:
        print(f"  - ERROR: An issue occurred while generating embeddings with the Hugging Face model. - {e}")
        return []

    return initial_candidates

# --- Stage 2: VLM Judgment using OpenAI's GPT-4o with Image Input ---

def build_vlm_prompt_messages(task_to_evaluate: dict, app_description: str, image_base64: str):
    system_prompt = """
    You are an expert AI assistant specializing in mobile GUI automation and testing. 
    Your mission is to determine if a given task is realistically executable on a specific application,
    by analyzing both its text description and a visual screenshot of its interface.
    """

    # user_prompt = f"""
    # ## Target Application Context
    # **Description:** {app_description}

    # ## Task to Evaluate
    # - **Task Description:** "{task_to_evaluate['instruction']}"
    # - **Macro Actions:** {task_to_evaluate['macro_actions']}

    # ## Your Task
    # Based on both the visual screenshot and the text description, evaluate if the task is feasible.

    # Respond ONLY with a valid JSON object in this format:
    # {{
    #     "is_relevant": true or false,
    #     "reason": "Short reason here."
    # }}
    # """
    
    user_prompt = f"""
    ## Target Application Context
    **Description:** {app_description}

    ## Task to Evaluate
    - **Task Description:** "{task_to_evaluate['instruction']}"
    - **Macro Actions:** {task_to_evaluate['macro_actions']}

    ## Your Task
    Based on the visual screenshot and the text descriptions, evaluate if the task is feasible on the given application screen.

    **Crucial Instruction:** You MUST ignore specific brand or website names  mentioned in the Task Description. Instead, focus on the core **intent** of the task (e.g., "shopping", "searching for an item"). Your judgment should be based on whether the **functionality** visible in the screenshot aligns with this core intent.

    Respond ONLY with a valid JSON object in this format:
    {{
        "is_relevant": true or false,
        "reason": "Short reason here."
    }}
    """
    
    return [
        {"role": "system", "content": system_prompt.strip()},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt.strip()},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_base64}"
                }
            ]
        }
    ]

def filter_by_vlm_judgment(
    candidate_tasks: list, 
    app_description: str, 
    image_path: str,
    client: OpenAI
) -> list:
    """
    Uses GPT-4o with vision to perform a deep, reasoned judgment on each candidate task.
    
    Note: Using gpt-4o as a substitute for the requested 'gpt-4.1', as it is the latest,
    most capable vision model available via the API.
    """
    final_tasks = []
    model_name = "gpt-4.1"
    print(f"\n[Stage 2] Starting deep-dive filtering with Vision LLM: '{model_name}'...")

    try:
        image_base64 = encode_image_to_base64(image_path)
    except Exception as e:
        print(f"  - ERROR: Could not read or encode the image at '{image_path}'. Skipping Stage 2. Error: {e}")
        return []

    for i, task in enumerate(candidate_tasks):
        print(f"\n[VLM Judgment {i+1}/{len(candidate_tasks)}] Evaluating Task ID: {task['id']}...")
        print(f"  - Task: {task['instruction']}")
        
        messages = build_vlm_prompt_messages(task, app_description, image_base64)
        
        try:
            response = client.responses.create(
                model=model_name,
                input=messages,
            )
            
            output_text = response.output_text.strip()
            if not output_text:
                raise ValueError("Empty output from VLM response")

            response_json = json.loads(response.output_text)
            
            is_relevant = response_json.get('is_relevant')
            reason = response_json.get("reason", "")
            
            print(f"  - VLM Verdict: {is_relevant}")
            print(f"  - Reason: {reason if reason else 'No reason provided'}")
            
            if is_relevant:
                final_tasks.append(task)
                
        except Exception as e:
            print(f"  - ERROR: An issue occurred during the VLM judgment call for Task ID {task['id']}. - {e}")
            continue
            
    return final_tasks

# --- Main Execution Block ---

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some integers.")
    # parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold for Stage 1 filtering')
    parser.add_argument('--agnostic_pool', type=str, default='gpt4_aitz_output.json', help='Path to the app-agnostic task pool JSON file')
    parser.add_argument('--app_description', type=str, default='google_map_description.txt')
    parser.add_argument("--app_image", type=str, default="/home/jhna/PAV/assets/google_maps.png")
    parser.add_argument("--output_path", type=str, default="google_map_pool.json", help="Path to save the output JSON file")
    parser.add_argument('--stage', type=int, choices=[1, 2], default=2, help='Stage of the pipeline to execute (1 or 2)')
    args = parser.parse_args()
    
    # client = OpenAI(api_key=args.api_key)
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    
    APP_IMAGE_PATH = args.app_image
    
    target_app_description_file = args.app_description
    with open(target_app_description_file, 'r', encoding='utf-8') as f:
        TARGET_APP_DESCRIPTION = f.read().strip()

    app_agnostic_pool = args.agnostic_pool
    with open(app_agnostic_pool, 'r', encoding='utf-8') as f:
        APP_AGNOSTIC_POOL = json.load(f)
        
    APP_AGNOSTIC_POOL = preprocess_json(APP_AGNOSTIC_POOL)
    
    if args.stage == 1:
        
        # The threshold might need tuning based on the model's behavior.
        SIMILARITY_THRESHOLD = args.threshold

        print("="*60)
        print("      Task Filtering Pipeline for Mobile GUI Automation")
        print("="*60)
        
        # --- Setup Stage 1: Load Hugging Face Model ---
        # Note: Using 'jina-embeddings-v2-base-en' as a substitute for the requested 'jina-embeddings-v3'.
        # This model is downloaded on first run and cached for later use.
        embedding_model_name = 'jinaai/jina-embeddings-v3'
        try:
            print(f"Loading Hugging Face embedding model: '{embedding_model_name}'...")
            embedding_model = SentenceTransformer(embedding_model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load Hugging Face model. Check your internet connection or model name. Error: {e}")
            exit()

        # --- Execute Stage 1 ---
        initial_candidates = filter_by_semantic_similarity(
            task_pool=APP_AGNOSTIC_POOL,
            app_description=TARGET_APP_DESCRIPTION,
            threshold=SIMILARITY_THRESHOLD,
            model=embedding_model
        )
        print(f"\nStage 1 Result: {len(initial_candidates)} of {len(APP_AGNOSTIC_POOL)} tasks selected as candidates.")
        print("Selected Candidate Task IDs:", [task['id'] for task in initial_candidates])
        
        print("\n" + "="*60 + "\n")
        
    else:
        initial_candidates = APP_AGNOSTIC_POOL

    final_selected_tasks = filter_by_vlm_judgment(
        candidate_tasks=initial_candidates, 
        app_description=TARGET_APP_DESCRIPTION, 
        image_path=APP_IMAGE_PATH,
        client=client
    )
    print("\n" + "="*60)
    print("                   Final Filtering Results")
    print("="*60)
    print(f"Out of {len(initial_candidates)} candidates, {len(final_selected_tasks)} tasks were ultimately selected.")
    # --- Save the final results to a JSON file ---
    if final_selected_tasks:
        output_filename = args.output_path
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(final_selected_tasks, f, indent=4, ensure_ascii=False)
            print(f"\nSuccessfully saved {len(final_selected_tasks)} approved tasks to '{output_filename}'")
        except Exception as e:
            print(f"\nError: Could not save the results to a file. - {e}")

        
