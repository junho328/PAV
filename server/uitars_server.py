import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import os

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
    
model_path = "ByteDance-Seed/UI-TARS-1.5-7B"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

os.makedirs("./uitars_data", exist_ok=True)

print("Model loaded successfully.")

app = FastAPI()

class Query(BaseModel):
    task: str
    image_base64: str
    step: int
    role: str
    
def parse_action_string(model_output) -> dict | None:
    """
    Parse a model output string containing an Action, e.g.
      "Thought: ...\nAction: click(start_box='(986,1899)')"
    Returns a dict like:
      {"action_type": "click", "start_box": (986, 1899)}
    or for argument‐less actions:
      {"action_type": "press_home"}
    """
    # 1) Pull out the “Action: …” portion
    m = re.search(r"Action:\s*(\w+\(.*\))", model_output)
    if not m:
        return None
    cmd = m.group(1)
    
    # 2) Split into action_type and inner args
    m2 = re.match(r"^(\w+)\((.*)\)$", cmd)
    if not m2:
        return None
    action_type, args_str = m2.group(1), m2.group(2).strip()
    
    result = {"action_type": action_type}
    if not args_str:
        return result
    
    # 3) Extract key='value' pairs
    for key, val in re.findall(r"(\w+)=['\"]([^'\"]*)['\"]", args_str):
        if key in ("start_box", "end_box", "coordinate"):
            # parse "(x,y)" → (int(x), int(y))
            nums = list(map(int, re.findall(r"-?\d+", val)))
            result[key] = tuple(nums)
        else:
            result[key] = val
    
    return result

@app.post("/predict")
def predict(query: Query):
    # 1) 입력 이미지 디코딩
    image_bytes = base64.b64decode(query.image_base64)
    
    os.makedirs("./uitars_screenshot", exist_ok=True)
    
    screenshot_path = f"./uitars_data/screenshot_{query.step}.png"
        
    with open(screenshot_path, "wb") as f:
        f.write(image_bytes)
        
    screenshot = Image.open(screenshot_path)
    
    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": screenshot,
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
    
    print(f">>output_text: {output_text}")
    
    # Qwen will perform action thought function call
    
    action_dict = parse_action_string(output_text[0])
    
    print(action_dict)
    
    # ex) {"name": "ui_tars", "arguments": {"action_type": "click", "start_box": [434, 226]}}
    response = {
        "name" : "ui_tars",
        "arguments": action_dict
    }
    
    return response

# uvicorn uitars_server:app --host 0.0.0.0 --port 8000