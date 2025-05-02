import subprocess
import argparse
import os
import sys
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from pav_agent.pav_agents import Planner, Actor, Verifier
plan = Planner()
act = Actor()
verify = Verifier()

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

import json
from PIL import Image

def get_screenshot(output_path):
    output_file = os.path.join(output_path, 'screenshot.png')
    subprocess.run(['adb', 'exec-out', 'screencap', '-p'], stdout=open(output_file, 'wb'))

def PAV(task_text, image_path):
    initial_screenshot = Image.open(image_path)
    user_query = 'instruction = "Please add Restaurant Yori to Want to go list."'

    previous_macro_action_plan = []
    # previous_macro_action_plan = [
    #                     "Search 'Namsan Tower'",
    #                     "Select direction",
    #                 ]

    task_completed = False
    step = 0    # macro action step
    while not task_completed:
        ## Planner Inference
        macro_action_plan = plan.planner(model, processor, initial_screenshot, user_query, previous_macro_action_plan)
        print(macro_action_plan)

        ## Actor Inference
        macro_mask = [False] * len(macro_action_plan)
        trial = 0   # micro action trial
        backup_screenshot_path = "curr_screenshot.png"
        initial_screenshot.save(backup_screenshot_path)
        curr_screenshot = "curr_screenshot.png"
        while not all(macro_mask):
            curr_macro_action = macro_action_plan[step]
            micro_action_plan = act.actor(model, processor, curr_screenshot, curr_macro_action)
            print(micro_action_plan)

            # Interact with Emulator
            action_type, action_coord = act.to_emulator(micro_action_plan)
            execute_action(action_type, action_coord)
            get_screenshot()
            trial += 1

            # Verifier Inference
            next_screenshot = 'screenshot.png'
            macro_mask[step] = verify.verifier(model, processor, curr_screenshot, next_screenshot, curr_macro_action)
            print("verification result : ", macro_mask[step])
            
            if macro_mask[step] == True:    # Achieve macro action
                step += 1
                next_screenshot.save(backup_screenshot_path)
                previous_macro_action_plan.append(curr_macro_action)
                trial = 0

            if trial > 5:                   # Fail macro action
                print("Macro action replan needed!")
                get_screenshot()
                initial_screenshot = 'screenshot.png'
                break
        
        if all(macro_mask):  # 모든 macro action 수행
            task_completed = True

    print("End of PAV")
    

def execute_action(action, arguments):
    if action == 'click':
        x = arguments[0]
        y = arguments[1]
        subprocess.run(['adb', 'shell', 'input', 'tap', str(x), str(y)])
    elif action == "swipe":
        x1 = arguments[0][0]
        y1 = arguments[0][1]
        x2 = arguments[1][0]
        y2 = arguments[1][1]
        subprocess.run(['adb', 'shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2)])
    elif action == "type":
        text = arguments
        subprocess.run(['adb', 'shell', 'input', 'text', text])
        
    print(f"Executed action: {action} with arguments: {arguments}")


def main(task_text, output_path):
    get_screenshot(output_path)


def vlm_pipeline(task_text, image_path):
    get_screenshot()
    PAV(task_text, 'screenshot.png')
    
if __name__ == "__main__":
    # 스크린샷을 찍고, 이미지를 처리하여 작업을 수행하는 파이프라인
    parser = argparse.ArgumentParser(description="VLM Pipeline")
    parser.add_argument('--task_text', type=str, required=True, help='Task text for the VLM pipeline')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save screenshots')
    args = parser.parse_args()
    
    # main(args.task_text, args.output_path)
    vlm_pipeline(args.task_text, args.output_path)
