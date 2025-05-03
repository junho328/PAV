import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaㄴseModel
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

import re

MOBILE_USE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
""" 

def parse_action_string(output):
    # 정규식 패턴 정의
    patterns = {
        'click': r"click\(start_box='<\|box_start\|\>\((\d+),(\d+)\)<\|box_end\|\>'\)",
        'long_press': r"long_press\(start_box='<\|box_start\|\>\((\d+),(\d+)\)<\|box_end\|\>'\)",
        'type': r"type\(content='(.*?)'\)",
        'scroll': r"scroll\(start_box='<\|box_start\|\>\((\d+),(\d+)\)<\|box_end\|\>', direction='(down|up|left|right)'\)",
        'open_app': r"open_app\(app_name='(.*?)'\)",
        'drag': r"drag\(start_box='<\|box_start\|\>\((\d+),(\d+)\)<\|box_end\|\>', end_box='<\|box_start\|\>\((\d+),(\d+)\)<\|box_end\|\>'\)",
        'press_home': r"press_home\(\)",
        'press_back': r"press_back\(\)",
        'finished': r"finished\(content='(.*?)'\)",
    }

    for action_type, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            if action_type in ['click', 'long_press']:
                return {'action_type': action_type, 'start_box': [int(match.group(1)), int(match.group(2))]}
            elif action_type == 'scroll':
                return {'action_type': action_type, 'start_box': [int(match.group(1)), int(match.group(2))], 'direction': match.group(3)}
            elif action_type == 'type':
                return {'action_type': action_type, 'content': match.group(1)}
            elif action_type == 'open_app':
                return {'action_type': action_type, 'app_name': match.group(1)}
            elif action_type == 'drag':
                return {
                    'action_type': action_type,
                    'start_box': [int(match.group(1)), int(match.group(2))],
                    'end_box': [int(match.group(3)), int(match.group(4))]
                }
            elif action_type in ['press_home', 'press_back']:
                return {'action_type': action_type}
            elif action_type == 'finished':
                return {'action_type': action_type, 'content': match.group(1)}
    
    # 매치되는 action이 없으면 None 반환
    return None

model_path = "ByteDance-Seed/UI-TARS-1.5-7B"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

print("Model loaded successfully.")

app = FastAPI()

class Query(BaseModel):
    task: str
    image_base64: str
    step: int

@app.post("/predict")
def predict(query: Query):
    # 1) 입력 이미지 디코딩
    image_bytes = base64.b64decode(query.image_base64)
    
    screenshot_path = f"./screenshot_{query.step}.png"
        
    with open(screenshot_path, "wb") as f:
        f.write(image_bytes)

    screenshot = Image.open(screenshot_path)
    
    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": screenshot_path,
                    },
                    {"type": "text", "text": MOBILE_USE.format(instruction=query.task)},
                ],
            }
        ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Qwen will perform action thought function call
    
    action_dict = parse_action_string(output_text[0])
    
    print(action_dict)
    
    # ex) {"name": "ui_tars", "arguments": {"action_type": "click", "start_box": [434, 226]}}
    response = {
        "name" : "ui_tars",
        "aguments": action_dict
    }
    
    return response

# uvicorn baseline_server:app --host 0.0.0.0 --port 8000 --workers 1