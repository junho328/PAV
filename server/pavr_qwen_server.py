import base64, io, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

import os
import shutil

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)

from qwen_vl_utils import smart_resize

from pav_agent.pavr_qwen_agents import Planner, Actor, Verifier

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from shot_composer import FewShotComposer

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
# model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

planner = Planner()
actor = Actor()
verifier = Verifier()

os.makedirs("./pavr_data", exist_ok=True)

app = FastAPI()

class Query(BaseModel):
    task: str
    image_base64: str
    step: int
    role: str
    previous_macro_action: str
    replan: str
    previous_action: str
    count: int
    threshold: int
    app_name: str

@app.post("/predict")    
def pavr(query: Query):
    
    user_query = query.task
    step = query.step
    
    if query.role == "planner":
        
        image_bytes = base64.b64decode(query.image_base64)
        screenshot_path = f"./pavr_data/screenshot_{step}.png"
        
        with open(screenshot_path, "wb") as f:
            f.write(image_bytes)
            
        if query.replan == "plan-initial":
            
            composer = FewShotComposer(app_name=query.app_name)
            composer.load("shot_pools")
            
            few_shots = composer.build_prompt(user_query, k=3)
            
            print(f">>>Few Shot Prompt:\n {few_shots}")

            macro_action_plan = planner.plan(
                model=model, 
                processor=processor, 
                task=user_query, 
                screenshot_path=screenshot_path, 
                app_name=query.app_name, 
                replan = query.replan,
                previous_macro_action=query.previous_macro_action,
                few_shots=few_shots
            )
            
            print(f">>>Planner Output: {macro_action_plan}")
            
            with open("./pavr_data/macro_plans.json", "w") as f:
                json.dump(macro_action_plan, f)
        
        elif query.replan == "replan-success":
            with open("./pavr_data/macro_plans.json", "r") as f:
                macro_action_plan = json.load(f)
            
            previous_macro_action = query.previous_macro_action.split(", ")
            
            macro_action_plan = planner.plan(
                model=model, 
                processor=processor,
                task=user_query,
                screenshot_path=screenshot_path,
                app_name=query.app_name,
                replan=query.replan,
                macro_action_list=macro_action_plan,
                previous_macro_action=previous_macro_action
            )
            
            print(f">>>Planner Output: {macro_action_plan}")
            
            with open("./pavr_data/macro_plans.json", "w") as f:
                json.dump(macro_action_plan, f)
        
        elif query.replan == "replan-fail":
            with open("./pavr_data/macro_plans.json", "r") as f:
                macro_action_plan = json.load(f)
            
            previous_macro_action = query.previous_macro_action.split(", ")[-1]
            
            macro_action_plan = planner.plan(
                model=model, 
                processor=processor,
                task=user_query,
                screenshot_path=screenshot_path,
                app_name=query.app_name,
                replan=query.replan,
                macro_action_list=macro_action_plan,
                previous_macro_action=previous_macro_action,
                step=step,
                threshold=query.threshold,
            )
            
            print(f">>>Planner Output: {macro_action_plan}")
            
            with open("./pavr_data/macro_plans.json", "w") as f:
                json.dump(macro_action_plan, f)
        
        else:
            raise NotImplementedError
                
        current_macro_action = macro_action_plan[0]
        
        micro_action = actor.act(
            model=model, 
            processor=processor, 
            macro_action_plan=macro_action_plan, 
            current_macro_action=current_macro_action, 
            screenshot_path=screenshot_path
        )

        print(">>>Actor Output:", current_macro_action, '\n', query.previous_action, '\n', micro_action["arguments"])
        # print(">>>Actor Output:", current_macro_action, '\n', micro_action["arguments"])

        # ex) {"name": "pavr_qwen", "arguments": {"action": "click", "coordinate": [935, 406]}}
        
        response = {
            "name": "pavr_qwen",
            "arguments": micro_action["arguments"],
            "macro_action_plan" : macro_action_plan,
            "current_macro_action": current_macro_action,
        }
    
    elif query.role == "actor":
        
        image_bytes = base64.b64decode(query.image_base64)
        screenshot_path = f"./pavr_data/screenshot_{step}.png"
        
        with open(screenshot_path, "wb") as f:
            f.write(image_bytes)
            
        with open("./pavr_data/macro_plans.json", "r") as f:
            macro_action_plan = json.load(f)
        
        current_macro_action = macro_action_plan[0]
        
        micro_action = actor.act(
            model=model, 
            processor=processor, 
            macro_action_plan=macro_action_plan,
            current_macro_action=current_macro_action, 
            screenshot_path=screenshot_path, 
            previous_micro_action=query.previous_action
        )

        action_type = micro_action["arguments"]["action"]
        
        print(">>>Actor Output:", current_macro_action, '\n', query.previous_action, '\n', micro_action["arguments"])
        
        if action_type == "terminate":
            macro_action_plan.pop(0)
            
            if len(macro_action_plan) == 0:
                print("All macro actions completed!")
                return {"name": "pavr_qwen", "arguments": {"action": "task_completed"}, "current_macro_action": current_macro_action}
            
            with open("./pavr_data/macro_plans.json", "w") as f:
                json.dump(macro_action_plan, f)

        # ex) {"name": "pavr_qwen", "arguments": {"action": "click", "coordinate": [935, 406]}}
        
        response = {
            "name": "pavr_qwen",
            "arguments": micro_action["arguments"],
            "current_macro_action": current_macro_action,
        }
    
    elif query.role == "verifier":
        
        with open("./pavr_data/macro_plans.json", "r") as f:
            macro_action_plan = json.load(f)
            
        current_macro_action = macro_action_plan[0]
        
        previous_screenshot_path = f"./pavr_data/screenshot_{step}.png"
        
        current_image_bytes = base64.b64decode(query.image_base64)
        
        with open(f"./pavr_data/verifier_screenshot_{step}.png", "wb") as f:
            f.write(current_image_bytes)
            
        current_screenshot_path = f"./pavr_data/verifier_screenshot_{step}.png"
        
        response = verifier.verify(
            model=model, 
            processor=processor, 
            macro_action=current_macro_action, 
            previous_screenshot_path=previous_screenshot_path, 
            current_screenshot_path=current_screenshot_path
        )

        verification  = response["task_completed"]
        
        print(f">>>Verifier Output: {response}")
        
        if verification:
            
            print(f"<{current_macro_action}> completed!")
            
            if len(macro_action_plan) > 1:
                macro_action_plan.pop(0)

                with open("./pavr_data/macro_plans.json", "w") as f:
                    json.dump(macro_action_plan, f)
                    
            else:
                
                return {"task_completed": -1 , "reason": "All macro actions are completed!", "current_macro_action": current_macro_action}
        else:
            if query.count > query.threshold - 1:
                macro_action_plan.pop(0)
                with open("./pavr_data/macro_plans.json", "w") as f:
                    json.dump(macro_action_plan, f)
            
            print(f"<{current_macro_action}> still in progress!")
            
    return response

# uvicorn pavr_qwen_server:app --host 0.0.0.0 --port 8000
