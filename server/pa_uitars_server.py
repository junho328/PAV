import base64, json
from fastapi import FastAPI
from pydantic import BaseModel

import os

from pav_agent.pav_uitars_agents import Planner, Actor

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "ByteDance-Seed/UI-TARS-1.5-7B"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

planner = Planner()
actor = Actor()

os.makedirs("./pa_uitars_data", exist_ok=True)

app = FastAPI()

class Query(BaseModel):
    task: str
    image_base64: str
    step: int
    role: str
    previous_action: str
    app_name: str

@app.post("/predict")    
def pa(query: Query):
    
    user_query = query.task
    step = query.step
    
    if query.role == "planner":
        
        image_bytes = base64.b64decode(query.image_base64)
        screenshot_path = f"./pa_uitars_data/screenshot_{step}.png"
        
        with open(screenshot_path, "wb") as f:
            f.write(image_bytes)

        macro_action_plan = planner.plan(model=model, processor=processor, task=user_query, screenshot_path=screenshot_path, app_name=query.app_name)
        
        print(f">>>Planner Output: {macro_action_plan}")
        
        with open("./pa_uitars_data/macro_plans.json", "w") as f:
            json.dump(macro_action_plan, f)
            
        current_macro_action = macro_action_plan[0]
        
        micro_action = actor.act(model=model, processor=processor, macro_action_plan=macro_action_plan, current_macro_action=current_macro_action, screenshot_path=screenshot_path)
        print(">>>Actor Output:", micro_action["arguments"])

        # ex) {"name": "pa_uitars", "arguments": {"action": "click", "coordinate": [935, 406]}}
        
        response = {
            "name": "pa_uitars",
            "arguments": micro_action["arguments"],
            "macro_action_plan" : macro_action_plan
        }
    
    elif query.role == "actor":
        
        image_bytes = base64.b64decode(query.image_base64)
        screenshot_path = f"./pa_uitars_data/screenshot_{step}.png"
        
        with open(screenshot_path, "wb") as f:
            f.write(image_bytes)
            
        with open("./pa_uitars_data/macro_plans.json", "r") as f:
            macro_action_plan = json.load(f)
        
        current_macro_action = macro_action_plan[0]
        
        micro_action = actor.act(model=model, processor=processor, macro_action_plan=macro_action_plan ,current_macro_action=current_macro_action, screenshot_path=screenshot_path, previous_micro_action=query.previous_action)
        action_type = micro_action["arguments"]["action"]
        
        print(">>>Actor Output:", micro_action["arguments"])
        
        if action_type == "terminate":
            macro_action_plan.pop(0)
            
            if len(macro_action_plan) == 0:
                print("All macro actions completed!")
                return {"name": "pa_qwen", "arguments": {"action": "task_completed"}}
            
            with open("./pa_uitars_data/macro_plans.json", "w") as f:
                json.dump(macro_action_plan, f)

        # ex) {"name": "pa_qwen", "arguments": {"action": "click", "coordinate": [935, 406]}}
        
        response = {
            "name": "pa_uitars",
            "arguments": micro_action["arguments"]
        }
            
    return response

# uvicorn pa_uitars_server:app --host 0.0.0.0 --port 8000