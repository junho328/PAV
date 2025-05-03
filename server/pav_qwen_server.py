import base64, io, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)

from qwen_vl_utils import smart_resize

from pav_agent.pav_agents import Planner, Actor, Verifier

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
#model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

planner = Planner()
actor = Actor()
verifier = Verifier()

app = FastAPI()

class Query(BaseModel):
    task: str
    image_base64: str
    step: int
    role: str

@app.post("/predict")    
def pav(query: Query):
    
    user_query = query.task
    step = query.step
    
    if query.role == "planner":
        
        image_bytes = base64.b64decode(query.image_base64)
        screenshot_path = f"./pav_data/screenshot_{step}.png"
        
        with open(screenshot_path, "wb") as f:
            f.write(image_bytes)
            
        screenshot = Image.open(screenshot_path)
        
        macro_action_plan = planner.planner(model, processor, user_query,screenshot)
        
        print(f">>>Planner Output: {macro_action_plan}")
        
        with open("./pav_data/macro_plans.json", "w") as f:
            json.dump(macro_action_plan, f)
            
        current_macro_action = macro_action_plan[0]
        
        micro_action = actor.actor(model, processor, screenshot, current_macro_action)
        print(">>>Actor Output:", micro_action["arguments"])

        # ex) {"name": "pav_qwen", "arguments": {"action": "click", "coordinate": [935, 406]}}
        
        response = {
            "name": "pav_qwen",
            "arguments": micro_action["arguments"],
            "macro_action_plan" : macro_action_plan
        }
    
    elif query.role == "actor":
        
        image_bytes = base64.b64decode(query.image_base64)
        screenshot_path = f"./pav_data/screenshot_{step}.png"
        
        with open(screenshot_path, "wb") as f:
            f.write(image_bytes)
            
        screenshot = Image.open(screenshot_path)
            
        with open("./pav_data/macro_plans.json", "r") as f:
            macro_action_plan = json.load(f)
        
        curr_macro_action = macro_action_plan[0]
        
        micro_action = actor.actor(model, processor, screenshot, curr_macro_action)
        print(">>>Actor Output:", micro_action["arguments"])

        # ex) {"name": "pav_qwen", "arguments": {"action": "click", "coordinate": [935, 406]}}
        
        response = {
            "name": "pav_qwen",
            "arguments": micro_action["arguments"]
        }
    
    elif query.role == "verifier":
        
        with open("./pav_data/macro_plans.json", "r") as f:
            macro_action_plan = json.load(f)
            
        current_macro_action = macro_action_plan[0]
        
        previous_screensho_path = f"./pav_data/screenshot_{step}.png"
        
        current_image_bytes = base64.b64decode(query.image_base64)
        
        with open(f"./pav_data/verifier_screenshot_{step}.png", "wb") as f:
            f.write(current_image_bytes)
            
        current_screenshot_path = f"./pav_data/verifier_screenshot_{step}.png"
        
        response = bool(verifier.verifier(model, processor, previous_screensho_path, current_screenshot_path, current_macro_action))
        
        if response == True:
            print(f">>>Verifier Output: <{current_macro_action}> Done!")
            macro_action_plan.pop(0)
            with open("./pav_data/macro_plans.json", "w") as f:
                json.dump(macro_action_plan, f)
        else:
            print(f">>>Verifier Output: <{current_macro_action}> still in progress!")
            
    return response

# uvicorn pav_qwen_server:app --host 0.0.0.0 --port 8000
