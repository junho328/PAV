import json
from PIL import Image

from qwen_vl_utils import smart_resize, process_vision_info

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)

from pav_agent.pav_agent_function_call import MobileUse

class Planner():
    
    def __init__(self):
        self.prompt =  """You are a helpful mobile agent and a good planner.
        You are given a screenshot of a mobile device and a task.
        You need to generate a macro action plan to complete the task.
        
        Example1:
        Task:
        Please display the route to KAIST College of Engineering.
        Macro Action Plan:
        [Search for KAIST College of Engineering, Select directions to KAIST College of Engineering, Display route]
        
        Example2:
        Task:
        Show me a wheelchair-accessible route to Gyeongbokgung Palace.
        Macro Action Plan:
        [Search for Gyeongbokgung Palace, Select directions to Gyeongbokgung Palace, Filter wheelchair-accessible route, Display route]
        
        Example3:
        Task:
        Please add restaurant Yori to Favorites list.
        Macro Action Plan:
        [Search for restaurant Yori, Select restaurant Yori, Add to Favorites list, Done actions]
        
        Example4:
        Task:
        Show me the cafes nearby Seoul Station.
        Macro Action Plan:
        [Search for Seoul Station, Set location to Seoul Station, Search for cafes, Display cafes]
        
        Now you are given a task and a screenshot.
        Generae a macro action plan to complete the task.
        
        Task: 
        {instruction}
        Macro Aciton Plan:
        """

    def plan(self, model, processor, task, screenshot_path):
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": screenshot_path,
                    },
                    {"type": "text", "text": self.prompt.format(instruction=task)},
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
        output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        raw = output[0]                    # ex: "[Search for Samsung Seoul R&D Campus, Select directions, Display route]"
        inner = raw.strip("[]")           # "Search for Samsung Seoul R&D Campus, Select directions, Display route"
        macro_action_plan = [item.strip() for item in inner.split(",")]
        
        return macro_action_plan
            
class Actor():
    def act(self, model, processor, current_macro_action, screenshot_path):
        
        user_query = f"Task to achieve: {current_macro_action}"

        dummy_image = Image.open(screenshot_path)
        resized_height, resized_width  = smart_resize(dummy_image.height,
            dummy_image.width,
            factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
            min_pixels=processor.image_processor.min_pixels,
            max_pixels=processor.image_processor.max_pixels,)
        mobile_use = MobileUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )
        
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
        inputs = processor(text=[text], images=[dummy_image], padding=True, return_tensors="pt").to('cuda')


        output_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        # Qwen will perform action thought function call
        action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        
        return action

class Verifier():
    
    def __init__(self):
        self.prompt = """You are a helpful mobile agent and a good verifier.
You are given two screenshots of a mobile device and a task.
Determine whether the task has been completed by examining the following two screenshots.
If the task has not been completed yet, return 0. If the task has been completed, return 1.

current task: {macro_action}

Think step by step and provide the final answer. And return the answer in the following format:
```json
{{
    "task_completed": 0 or 1,
    "reason": "reason why the current task is completed or not"
}}
```
"""
      
    def verify(self, model, processor, macro_action, previous_screenshot_path, current_screenshot_path):
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": previous_screenshot_path},
                    {"type": "image", "image": current_screenshot_path},
                    {"type": "text",  "text": self.prompt.format(macro_action=macro_action)},
                ],
            }
        ]

        # 4. Apply the chat template to get the prompt text
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 5. Extract vision inputs (this will download & preprocess both images)
        image_inputs, video_inputs = process_vision_info(messages)

        # 6. Run through the processor (batch of 1 example with 2 images)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # 7. Generate and decode
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        # strip off the prompt tokens
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Extract Verification Information
        verifier_output = json.loads(output_text.split('```json')[1].split('```')[0])

        return verifier_output
