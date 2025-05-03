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
from PAV.server.agent_function_call import MobileUse

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

app = FastAPI()

class Query(BaseModel):
    task: str
    image_base64: str
    step: int

@app.post("/predict")
def predict(query: Query):
    # 1) 입력 이미지 디코딩
    image_bytes = base64.b64decode(query.image_base64)
    screenshot = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # The operation history can be orgnized by Step x: [action]; Step x+1: [action]...
    user_query = f'The user query: {query.task}'

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
    inputs = processor(text=[text], images=[screenshot], padding=True, return_tensors="pt").to('cuda')


    output_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(output_text)

    # Qwen will perform action thought function call
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])

    # ex) {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [935, 406]}}
    response = action['arguments']
    
    return response
