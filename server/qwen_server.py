import base64, io, json, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)

from qwen_vl_utils import smart_resize
from agent_function_call import MobileUse

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
#model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

os.makedirs("./qwen_data", exist_ok=True)

app = FastAPI()

class Query(BaseModel):
    task: str
    image_base64: str
    step: int
    role: str

@app.post("/predict")
def predict(query: Query):
    # 1) 입력 이미지 디코딩
    image_bytes = base64.b64decode(query.image_base64)
    
    screenshot_path = f"./qwen_data/screenshot_{query.step}.png"
        
    with open(screenshot_path, "wb") as f:
        f.write(image_bytes)
        
    screenshot = Image.open(screenshot_path)

    # The operation history can be orgnized by Step x: [action]; Step x+1: [action]...
    user_query = f'''Given a user query, You have to perform the task in a mobile environment.
First, decompose the task into high-level macro actions.
Second, execute micro actions step by step to achieve each macro action.(Micro actions: key, click, long_press, swipe, type, system_button, open, wait, terminate.)
Third, after executing each micro action, verify whether the macro action has been successfully completed.(Multiple micro actions may be required to accomplish a single macro action.)

The user query: {query.task}'''

    # The resolution of the device will be written into the system prompt. 
    resized_height, resized_width  = smart_resize(screenshot.height,
        screenshot.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,)
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
    )

    # Build messages
    
    prompt = NousFnCallPrompt()
    raw_messages = [
            Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
            Message(role="user", content=[
                ContentItem(text=user_query),
                ContentItem(image=f"file://{screenshot_path}")
            ]),
        ]

    message_objs = prompt.preprocess_fncall_messages(
        messages=raw_messages,
        functions=[mobile_use.function],
        lang=None,
    )
    
    message = [msg.model_dump() for msg in message_objs]
    
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[screenshot], padding=True, return_tensors="pt").to('cuda')


    output_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(output_text)

    # Qwen will perform action thought function call
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])

    # ex) {"name": "qwen", "arguments": {"action": "click", "coordinate": [935, 406]}}
    
    response = {
        "name" : "qwen",
        "arguments": action["arguments"]
    }
    
    return response

# uvicorn qwen_server:app --host 0.0.0.0 --port 8000
