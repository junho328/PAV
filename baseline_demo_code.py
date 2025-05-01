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

def get_running_emulator_ports():
    output = subprocess.check_output(['adb', 'devices']).decode()
    lines = output.strip().split('\n')[1:]  # skip header
    ports = []
    for line in lines:
        if line.startswith('emulator-'):
            port = int(line.split()[0].split('-')[1])
            ports.append(port)
    return ports

def wait_for_emulator(emulator_name, timeout=60):
    """ADB에서 에뮬레이터가 'device' 상태가 될 때까지 대기"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            output = subprocess.check_output(['adb', '-s', emulator_name, 'shell', 'getprop', 'sys.boot_completed'])
            if output.strip() == b'1':
                print(f"{emulator_name} is ready.")
                return True
        except subprocess.CalledProcessError:
            pass
        time.sleep(2)
    raise TimeoutError(f"Emulator {emulator_name} did not boot in time.")

def start_emulator(args):
    sdk_root = os.environ.get("ANDROID_SDK_ROOT")
    if not sdk_root:
        raise RuntimeError("환경 변수 ANDROID_SDK_ROOT 가 설정되어 있지 않습니다.")
    emulator_bin = os.path.join(sdk_root, "emulator", "emulator")

    # 기존 포트 목록 확인 및 다음 포트 결정
    used_ports = get_running_emulator_ports()
    base_port = 5554
    while base_port in used_ports:
        base_port += 2  # 2단위로 증가

    args.emulator_port = base_port  # 추가 인자로 저장해두기
    emulator_name = f"emulator-{base_port}"

    cmd = [
        emulator_bin,
        "-avd", args.avd_name,
        "-port", str(base_port),
        "-snapshot", args.snapshot,
        "-no-skin",
        "-no-window",
        "-no-audio",
        "-gpu", "swiftshader_indirect"
    ]

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"Starting emulator {args.avd_name} at port {base_port}…")

    # 에뮬레이터 부팅 완료까지 기다림
    wait_for_emulator(emulator_name)

    subprocess.run(["adb", "-s", emulator_name, "emu", "geo", "fix", str(127.0283), str(37.4672)], check=True)
    
def get_screenshot(image_path, emulator_name):
    subprocess.run(['adb', '-s', emulator_name, 'exec-out', 'screencap', '-p'], stdout=open(image_path, 'wb'))

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
    
    elif action_type == "system_button":
        
        button = action["arguments"]["button"]
        
        return "system_button", button
    
    elif action_type == "terminate":
        status = action["arguments"]["status"]
        
        return "terminate", status
    
    else:
        print(f"Model action output: {action}")
        subprocess.run(['adb', 'emu', 'kill'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise ValueError(f"Unknown action type: {action_type}")

def execute_action(action, arguments, emulator_name):
    base_cmd = ['adb', '-s', emulator_name, 'shell', 'input']
    if action == 'click':
        subprocess.run(base_cmd + ['tap', str(arguments[0]), str(arguments[1])])
    elif action == "swipe":
        subprocess.run(base_cmd + ['swipe', str(arguments[0][0]), str(arguments[0][1]), str(arguments[1][0]), str(arguments[1][1])])
    elif action == "type":
        text = arguments.replace(" ", "%s").replace("&", r"\&")
        subprocess.run(base_cmd + ['text', text])
    elif action == "system_button":
        button = arguments
        if button == "Back":
            subprocess.run(base_cmd + ['keyevent', '4'])
        elif button == "Home":
            subprocess.run(base_cmd + ['keyevent', '3'])
        elif button == "Menu":
            subprocess.run(base_cmd + ['keyevent', '82'])
        elif button == "Enter":
            subprocess.run(base_cmd + ['keyevent', '66'])
    print(f"Executed action: {action} with arguments: {arguments}")
    
def main(args):
    start_emulator(args)
    emulator_name = f"emulator-{args.emulator_port}"
    
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    os.makedirs(args.output_path, exist_ok=True)

    for i in range(args.max_steps):    
        image_path = os.path.join(args.output_path, f"step_{i}_screenshot.png")
        get_screenshot(image_path, emulator_name)
        print(f">>>> Saved screenshot\n")
        time.sleep(3)
        time.sleep(3)

        action, arguments = get_action(model, processor, args.task_text, image_path)
        print(f">>>> <Model Output> # Step {i+1}: # Action={action}, # Args={arguments}\n")

        if action == "terminate":
            print(">>>>>Terminating the process.")
            break

        execute_action(action, arguments, emulator_name)
        time.sleep(3)

    get_screenshot(os.path.join(args.output_path, "step_10_screenshot.png"), emulator_name)
    print(f">>>>> Process completed within {i+1} steps.\n")

    print(">>>>> Shutting down emulator…")
    subprocess.run(['adb', '-s', emulator_name, 'emu', 'kill'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
if __name__ == "__main__":
    # 스크린샷을 찍고, 이미지를 처리하여 작업을 수행하는 파이프라인
    
    parser = argparse.ArgumentParser(description="Qwen Moblie Agent Pipeline")
    parser.add_argument('--task_text', type=str, required=True, help='Task text for the VLM pipeline')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save screenshots')
    parser.add_argument('--avd_name', type=str, default="pixel7_api36_google", help='Name of the AVD to start')
    parser.add_argument('--snapshot', type=str, default="google_login", help='Snapshot name for the AVD')
    parser.add_argument('--max_steps', type=int, default=10, help='Maximum number of steps to execute')
    
    args = parser.parse_args()
    
    main(args)
