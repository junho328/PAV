import os
import json
from glob import glob
import random
import base64
from openai import OpenAI
import json
from pathlib import Path
import argparse

SYSTEM_PROMPT = """You are an expert mobile-UI analyst.  
Given:  
1. a high-level task in English, and  
2. a *chronologically ordered* list of screenshots showing how an expert completed that task,  

You should infer the minimal, ordered list of **macro actions** required — **ending with the step that shows the final target screen requested in the Task**.  
Macro actions are high-level actions that a user can take in mobile device.
The macro actions should be ordered in a way that reflects the sequence of actions taken in the screenshots.

Below is the example of macro actions:
- Search for <TARGET>  
- Select <TARGET>          
- Filter by <CRITERION>
- Show <TARGET>        
- Add <TARGET1> to <TARGET2>  
- Remove <TARGET> from cart
...
You can make up macro actions that are not listed above, but they should be high-level and relevant to the task.

Respond **only** in valid JSON with exactly the following format:  
{"macro_actions": [string]}  

Below are some examples of the macro actions and their indexes (note that each list ends by showing the requested target screen).
---
Example 1  
Input Task: Please display the route to Yangjae station.  
Output:  
{"macro_actions":[Search for Yangjae station, Show routes]}

Example 2  
Input Task: Add Shin Ramyeon to cart.  
Output:  
{"macro_actions":[Search for Shin Ramyeon,Select Shin Ramyeon,Add Shin Ramyeon to cart]}
---

"""

USER_PROMPT   = """
Now infer the macro actions and their indexes for the following task and screenshots.

Input Task: {instruction}
Output:
"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main(args):

    base_dir = args.base_dir
    subsets = ["general", "google_apps", "web_shopping"]

    general_data= []
    google_apps_data = []
    web_shopping_data = []

    for subset in subsets:
        subset_path = os.path.join(base_dir, subset)
        for folder_name in os.listdir(subset_path):
            folder_path = os.path.join(subset_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # JSON 파일 읽기
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            if not json_files:
                continue

            json_path = os.path.join(folder_path, json_files[0])
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    instruction = json_data[0]["instruction"]
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue

            # 이미지 파일 경로 수집 (확장자 제한 가능)
            image_files = sorted(glob(os.path.join(folder_path, '*.[jp][pn]g')))
            
            if subset == "general":
                general_data.append({
                    'folder': folder_name,
                    'instruction': instruction,
                    'image_paths': image_files
                })
            elif subset == "google_apps":
                google_apps_data.append({
                    'folder': folder_name,
                    'instruction': instruction,
                    'image_paths': image_files
                })
            elif subset == "web_shopping":
                web_shopping_data.append({
                    'folder': folder_name,
                    'instruction': instruction,
                    'image_paths': image_files
                })    
                
    random.seed(42)
    for data in [general_data, google_apps_data, web_shopping_data]:
        random.shuffle(data)
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    client = OpenAI(api_key=api_key)
    subsets = ["general", "google_apps", "web_shopping"]

    idx = 0
    gpt_response = {}
    for subset_data in [general_data, google_apps_data, web_shopping_data]:
        
        for task in subset_data[:100]:
               
            subset =  subsets[idx]
            task_id = task["folder"].split("-")[-1]
            instruction = task["instruction"]
            images = task["image_paths"]

            image_contents = [
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encode_image(img)}",
                }
                for img in images
            ]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "input_text", "text": USER_PROMPT.format(instruction=instruction)}] + image_contents}
            ]

            try:
                response = client.responses.create(
                    model="gpt-4.1",
                    input=messages,
                )
                output = response.output_text
            except Exception as e:
                output = f"Error: {str(e)}"

            print(f"[{subset}/{task_id}]")
            print(f"Task: {instruction}")
            print(output)
            print("-" * 50)

            if subset not in gpt_response:
                gpt_response[subset] = {}
                
            output_dict = {"instruction": instruction, "macro_actions": output}
            gpt_response[subset][task_id] = output_dict
            
        idx += 1
        
    with open(os.path.join(args.output_path), 'w', encoding='utf-8') as f:
        json.dump(gpt_response, f, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/home/jhna/PAV/android_in_the_zoo/train', help='Base directory for the dataset')
    parser.add_argument("--output_path", type=str, default="gpt4_aitz_output.json", help="Path to save the output JSON file")
    args = parser.parse_args()

    main(args)
    
