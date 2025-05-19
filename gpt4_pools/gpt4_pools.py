from pathlib import Path
import base64
from openai import OpenAI
import json
import argparse

# YOU SHOULD MAKE EXPERT SCREENSHOTS BEFORE RUNNING THIS CODE

SYSTEM_PROMPT = """You are an expert mobile-UI analyst.  
Given:  
1. a high-level task in plain English, and  
2. a *chronologically ordered* list of screenshots showing how an expert completed that task,  

infer the minimal, ordered list of **macro actions** required — **ending with the step that shows the final target screen requested in the Task**.  
Macro actions must be chosen from the ID catalogue below.  
Respond **only** in valid JSON with exactly two keys:  
{"indexes": [int], "macro_actions": [string]}  
Never add extra keys, text, or explanations.

<Macro-Action Catalogue>
1 : Search for <TARGET>  
2 : Select <TARGET>            (tap a result / UI element)  
3 : Filter by <CRITERION>      (e.g. price, popularity)  
4 : Show <TARGET>              (show navigation options)  
5 : Add <TARGET> to cart  
6 : Remove <TARGET> from cart  
---

Below are some examples of the macro actions and their indexes (note that each list ends by showing the requested target screen).
---
Example 1  
Input Task: Please display the route to Yangjae station.  
Output:  
{"indexes":[1,4],
"macro_actions":[Search for Yangjae station"Show routes]}

Example 2  
Input Task: Add Shin Ramyeon to cart.  
Output:  
{"indexes":[1,2,5],
"macro_actions":"Search for Shin Ramyeon,Select Shin Ramyeon,Add Shin Ramyeon to cart]}

Example 3  
Input Task: Please display a wheelchair-accessible route to Seoul station.  
Output:  
{"indexes":[1,4,3,4],
"macro_actions":[Search for Seoul station,Show routes,Filter by wheelchair-accessible route,Show filtered routes]}

Example 4  
Input Task: Show me the current rating of restaurant Yori.  
Output:  
{"indexes":[1,2,4],
"macro_actions":[Search for restaurant Yori,Select Yori,Show rating]}  
---

"""

USER_PROMPT   = """
Now infer the macro actions and their indexes for the following task and screenshots.

Input Task: {task}
Output:
"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def make_image_dict(args):
    
    root_folder = Path(args.root_folder) # "/home/jhna/shots_image"

    all_images = {}

    # app_list = ["google_maps", "aliexpress"]
    for app in args.app_list:
        app_folder = root_folder / app
        if not app_folder.is_dir():
            continue
        all_images[app] = {}

        # 각 task 폴더 순회
        for task_folder in app_folder.iterdir():
            if not task_folder.is_dir() or task_folder.name.startswith("."):
                continue
            task_name = task_folder.name
            all_images[app][task_name] = {}

            # 먼저 이 task 폴더 안에 서브폴더가 있는지 확인
            subdirs = [p for p in task_folder.iterdir() if p.is_dir() and not p.name.startswith(".")]

            if subdirs:
                # 서브폴더별로 PNG 수집
                for subtask_folder in subdirs:
                    images = [
                        p for p in subtask_folder.iterdir()
                        if p.is_file() and p.suffix.lower() == ".png"
                    ]
                    images_sorted = sorted(images, key=lambda p: int(p.stem))
                    all_images[app][task_name][subtask_folder.name] = [str(p) for p in images_sorted]

            else:
                # 서브폴더가 없으면, task 폴더 내 PNG들을 subtask '0' 으로 처리
                images = [
                    p for p in task_folder.iterdir()
                    if p.is_file() and p.suffix.lower() == ".png"
                ]
                images_sorted = sorted(images, key=lambda p: int(p.stem))
                all_images[app][task_name]["0"] = [str(p) for p in images_sorted]
    
    return all_images 

def main(args):
    
    client = OpenAI(api_key=args.api_key)
    
    with open(args.task_dict_path, "r") as f:
        task_dict = json.load(f)
    
    all_images = make_image_dict(args)
    
    apps = args.app_list
    
    all_response = {app: {} for app in apps}

    for app in apps:
        for task_id, task_name in task_dict[app].items():
            
            imgs = all_images[app][task_id]["0"]
            
            image_contents = [
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encode_image(img)}",
                    "detail" : "high"
                }
                for img in imgs
            ]
            
            message = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "input_text", "text": USER_PROMPT.format(task=task_name)}] + image_contents}
            ]

            response = client.responses.create(
                model="gpt-4.1",
                input=message
            )
            
            all_response[app][task_id] = response
            print(f"Task: {task_name}")
            print(f"Response: {response.output_text}")
            print("-" * 50)
            
            
    gpt_response  = {}
    for key, value in all_response.items():
        
        task_response = {}
        print(f"App: {key}")
        for task_id, response in value.items():
            print(f"Task ID: {task_id}")
            print(f"Response: {response.output_text}")
            print("-" * 50)
            
            task_response[task_id] = response.output_text
            
        gpt_response[key] = task_response

    import json 
    with open(args.output_path+"gpt4_pools.json", "w") as f:
        json.dump(gpt_response, f, indent=4)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="GPT-4 Pooling")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing images")
    parser.add_argument("--task_dict_path", type=str, required=True, help="Path to the task dictionary JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the JSON file")
    parser.add_argument("--app_list", type=str, nargs="+", required=True, help="List of apps to process") # python gpt4_pools.py --app_list google_maps aliexpress
    args = parser.parse_args()
    
    main(args)

## Example of task_dict
# task_dict = {
#     "google_maps": {
#         "0": "Please display the route to Seoul station.",
#         "1": "Show me a wheelchair-accessible route to Seoul station.",
#         "2": "Show me the current rating of cafe SimJae.",
# "3" : "Please add cafe SimJae to Want to go list.",
# "4" : "Show me the cafes nearby Yangjaecheon.",
# "5" : "Show me the opening hours of cafe SimJae.",
# "6" : "Show me the popular visiting times for Deoksugung Palace.",
# "7" : "Switch to satellite view of the map.",
# "8" : "Show me the photos of Seoul Forest Park.",
# "9" : "Show me the phone number of cafe SimJae.",
# "10" : "Show me the reviews of cafe SimJae.",
# "11" : "Show me the satellite view of Bukchon Hanok Village.",
# "12" : "Remove cafe Simjae from my Want to go list.",
# "13" : "Check the notifications in updates."
#     },
#     "aliexpress":{
#         "0":"Add samdasoo to cart.",
# "1":"Please buy fanta.",
# "2":"Add the most popular mouse pad to cart.",
# "3":"Add the cheapest usb-c cable to cart.",
# "4":"Show me the delivered items in my orders.",
# "5":"Show me the wishlist in my account.",
# "6":"Remove second item in my cart.",
# "7":"Show me the bluetooth speaker on sale.",
# "8":"Empty the cart.",
# "9":"View the items in New Arrivals of K-VENUE page."
#     }
# }
