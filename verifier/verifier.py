import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    Message,
    ContentItem,
)
import json
from PIL import Image

model_path3 = "Qwen/Qwen2.5-VL-3B-Instruct"
model_path7 = "Qwen/Qwen2.5-VL-7B-Instruct"
model3 = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path3, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
# model7 = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path7, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path3)

def verifier(model, screenshot1, screenshot2, action):
    user_query = f"""
Determine whether the action has been completed by examining the following two screenshots.
If the action has not been completed yet, return 0. If the action has been completed, return 1.
Action: {action}

Think step by step and provide the final answer. And return the answer in the following format:
<verify>
{{
    "action_completed": 0,
    "reason": "The action has not been completed yet."
}}
</verify>
    """

    # The resolution of the device will be written into the system prompt. 
    dummy_image1 = Image.open(screenshot1)
    dummy_image2 = Image.open(screenshot2)

    message = [
        Message(role="system", content=[ContentItem(text="You are a helpful mobile agent and a good verifier")]),
        Message(role="user", content=[
            ContentItem(text=user_query),
            ContentItem(image=f"file://{screenshot1}"),
            ContentItem(image=f"file://{screenshot2}")
        ]),
    ]
    message = [msg.model_dump() for msg in message]

    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    print("text",text)
    inputs = processor(text=[text], images=[dummy_image1, dummy_image2], padding=True, return_tensors="pt").to('cuda')


    output_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print('output')
    print(output_text, '\n')

    # Qwen will perform action thought function call
    action = json.loads(output_text.split('<verify>\n')[1].split('\n</verify>')[0])
    print(f"verify: {action['action_completed']}")
    print(f"reason: {action['reason']}")