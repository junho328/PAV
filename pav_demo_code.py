import subprocess
import argparse
import os
import sys
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from qwen_vl_utils import smart_resize
import json
from PIL import Image
from utils.agent_function_call import MobileUse

def get_screenshot(output_path):
    output_file = os.path.join(output_path, 'screenshot.png')
    subprocess.run(['adb', 'exec-out', 'screencap', '-p'], stdout=open(output_file, 'wb'))

def get_action(task_text, image_path):

    image = Image.open(image_path)
    user_query = 'The user query:  Open the file manager app and view the au_uu_SzH3yR2.mp3 file in MUSIC Folder\nTask progress (You have done the following operation on the current device): Step 1: {"name": "mobile_use", "arguments": {"action": "open", "text": "File Manager"}}; '
    
    resized_height, resized_width  = smart_resize(dummy_image.height,
        dummy_image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,)
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
    )
    
    message = NousFnCallPrompt.preprocess_fncall_messages(
        messages = [
            Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
            Message(role="user", content=[
                ContentItem(text=user_query),
                ContentItem(image=f"file://{screenshot}")
            ]),
        ],
        functions=[mobile_use.function],
        lang=None,
    )
    message = [msg.model_dump() for msg in message]
    
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    print("text",text)
    inputs = processor(text=[text], images=[dummy_image], padding=True, return_tensors="pt").to('cuda')
    
    output_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(output_text)

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
    action, coords = get_action(task_text, 'screenshot.png')
    execute_action(action, coords)
    
if __name__ == "__main__":
    # 스크린샷을 찍고, 이미지를 처리하여 작업을 수행하는 파이프라인
    
    parser = argparse.ArgumentParser(description="VLM Pipeline")
    parser.add_argument('--task_text', type=str, required=True, help='Task text for the VLM pipeline')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save screenshots')
    args = parser.parse_args()
    
    main(args.task_text, args.output_path)
