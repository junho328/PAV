import subprocess
import argparse
import os
import sys
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import time

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from qwen_vl_utils import smart_resize
import json
from PIL import Image
from agent_function_call import MobileUse

def start_emulator(avd_name="pixel7_api36_google", snapshot="map"):
    """
    ANDROID_SDK_ROOT/emulator/emulator 를 이용해 AVD를 백그라운드로 실행
    """
    sdk_root = os.environ.get("ANDROID_SDK_ROOT")
    if not sdk_root:
        raise RuntimeError("환경 변수 ANDROID_SDK_ROOT 가 설정되어 있지 않습니다.")
    emulator_bin = os.path.join(sdk_root, "emulator", "emulator")
    cmd = [
        emulator_bin,
        "-avd", avd_name,
        "-snapshot", snapshot,
        "-no-skin",
        "-no-window",
        "-no-audio",
        "-gpu", "swiftshader_indirect"
    ]
    # 백그라운드 실행
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # subprocess.run(['adb', 'exec-out', 'screencap', '-p'], stdout=open(image_path, 'wb'))
    # 에뮬레이터가 완전히 부팅될 때까지 충분히 대기
    print(f"Starting emulator {avd_name}…")
    time.sleep(15)
    
    # set location
    subprocess.run(["adb","emu","geo","fix", str(127.0283), str(37.4672)], check=True)
    
def get_screenshot(image_path):
    subprocess.run(['adb', 'exec-out', 'screencap', '-p'], stdout=open(image_path, 'wb'))

def get_action(model, processor, task_text, image_path):

    image = Image.open(image_path)
    user_query = f"The user query: {task_text}"
    
    resized_height, resized_width  = smart_resize(image.height,
        image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,)
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
    )
    
    prompt = NousFnCallPrompt()
    raw_messages = messages = [
            Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
            Message(role="user", content=[
                ContentItem(text=user_query),
                ContentItem(image=f"file://{image_path}")
            ]),
        ]

    message_objs = prompt.preprocess_fncall_messages(
        messages=raw_messages,
        functions=[mobile_use.function],
        lang=None,
    )
    
    message = [msg.model_dump() for msg in message_objs]
    
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    # print("text",text)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')
    
    output_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    # print(output_text)

    # Qwen will perform action thought function call
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
    
    action_type = action["arguments"]["action"]
    
    if action_type == "click":
        coordinate = action["arguments"]["coordinate"]
        
        return "click", coordinate
    
    elif action_type == "swipe":
        coordinate1 = action["arguments"]["coordinate"]
        coordinate2 = action["arguments"]["coordinate2"]
    
        return "swipe", (coordinate1, coordinate2)
    
    elif action_type == "type":
        text = action["arguments"]["text"]
        
        return "type", text
    
    elif action_type == "terminate":
        status = action["arguments"]["status"]
        
        return "terminate", status

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
        text = arguments.replace(" ", "%s")
        subprocess.run(['adb', 'shell', 'input', 'text', text])
        
    print(f"Executed action: {action} with arguments: {arguments}")
    
def main(task_text, output_path):
    
    start_emulator()
    
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)
    
    os.makedirs(output_path, exist_ok=True)
    
    for i in range(10):    
        
        image_path = os.path.join(output_path, f"step_{i}_screenshot.png")
        
        get_screenshot(image_path)
        time.sleep(3)
        
        action, arguments = get_action(model, processor, task_text, image_path)
        
        print(f"<Model Output> # Step {i+1}: # Action={action}, # Args={arguments}")
        
        if action == "terminate":
            print("Terminating the process.")
            break
        
        execute_action(action, arguments)
        
        time.sleep(3)
        
    print(f"Process completed within {i+1} steps.")
    
    # 3) 에뮬레이터 종료
    print("Shutting down emulator…")
    # adb 명령으로 graceful하게 종료
    subprocess.run(['adb', 'emu', 'kill'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 프로세스가 살아있다면 강제 종료
    emulator_proc.terminate()
    emulator_proc.wait(timeout=5)
    
if __name__ == "__main__":
    # 스크린샷을 찍고, 이미지를 처리하여 작업을 수행하는 파이프라인
    
    parser = argparse.ArgumentParser(description="Qwen Moblie Agent Pipeline")
    parser.add_argument('--task_text', type=str, required=True, help='Task text for the VLM pipeline')
    parser.add_argument('--output_path', type=str, default="qwen_screenshot/", help='Path to save screenshots')
    args = parser.parse_args()
    
    main(args.task_text, args.output_path)
