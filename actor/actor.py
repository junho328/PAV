from PIL import Image
from PIL import Image, ImageDraw, ImageColor
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from qwen_vl_utils import smart_resize
import json
from PIL import Image
from agent_function_call import MobileUse

from IPython.display import display


def draw_point(image: Image.Image, point: list, color=None):
    from copy import deepcopy
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)  
        except ValueError:
            color = (255, 0, 0, 128)  
    else:
        color = (255, 0, 0, 128)  
 
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = min(image.size) * 0.05
    x, y = point

    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color  # Red with 50% opacity
    )

    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)

    return combined.convert('RGB')

model_path3 = "Qwen/Qwen2.5-VL-3B-Instruct"
model_path7 = "Qwen/Qwen2.5-VL-7B-Instruct"
model3 = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path3, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
# model7 = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path7, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path3)

def actor(model, screenshot, user_query, display=True):
    # The resolution of the device will be written into the system prompt. 
    dummy_image = Image.open(screenshot)
    resized_height, resized_width  = smart_resize(dummy_image.height,
        dummy_image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,)
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
    )

    # Build messages
    message = NousFnCallPrompt().preprocess_fncall_messages(
        messages = [
            Message(role="system", content=[ContentItem(text="You are a helpful mobile agent.")]),
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
    print('output')
    print(output_text)

    # Qwen will perform action thought function call
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
    print("action")
    print(action)

    # As an example, we visualize the "click" action by draw a green circle onto the image.
    if display:
        display_image = dummy_image.resize((resized_width, resized_height))
        if action['arguments']['action'] == "click" or action['arguments']['action'] == "left_click":
            display_image = draw_point(dummy_image, action['arguments']['coordinate'], color='green')
            display(display_image)
        else:
            display(display_image)