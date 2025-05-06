import re
import json
from PIL import Image

from qwen_vl_utils import smart_resize, process_vision_info

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    Message,
    ContentItem,
)

class Planner():
    
    def __init__(self):
        self.prompt =  """You are a helpful mobile agent and a good planner.
        You are given a screenshot of a mobile device and a task.
        You need to generate a macro action plan to complete the task.
        
        Below are some examples of tasks and their corresponding macro action plans.
        ---
        <Example 1>
        Task:
        Please display the route to Jejujib.
        Macro Action Plan:
        [Search for Jejujib, Show the routes, Display the directions]
        
        <Example 2>
        Task:
        Please display the route to Yori.
        Macro Action Plan:
        [Search for Yori, Show the routes, Display the directions]
        
        <Example 3>
        Task:
        Show me a wheelchair-accessible route to Seoul Forest Park.
        Macro Action Plan:
        [Search for Seoul Forest Park, Show the routes, Display the directions, Filter by wheelchair-accessible route, Show the filtered route]
        
        <Example 4>
        Task:
        Show me a less walking route to Gangnam station.
        Macro Action Plan:
        [Search for Gangnam station, Show the routes, Display the directions, Filter by less walking route, Show the filtered route]
        
        <Example 5>
        Task:
        Show me a fewer transfers route to Lotte Tower.
        Macro Action Plan:
        [Search for Lotte Tower, Show the routes, Display the directions, Filter by fewer transfers route, Show the filtered route]
        
        <Example 6>
        Task:
        Please add restaurant Gimgane to Favorites list.
        Macro Action Plan:
        [Search for restaurant Gimgane, Add Gimgane to Favorites list]
        
        <Example 7>
        Task:
        Please add cafe Simjae to Want to go list.
        Macro Action Plan:
        [Search for cafe Simjae, Add Simjae to Want to go list]
        
        <Example 8>
        Task:
        Show me the current rating of restaurant Sushiyoung.
        Macro Action Plan:
        [Search for restaurant Sushiyoung, Select restaurant Sushiyoung, Check the rating section]
        
        <Example 9>
        Task:
        Show me the restaurants nearby KAIST College of Business.
        Macro Action Plan:
        [Search for KAIST College of Business, Select KAIST College of Business, Search for restaurants, Show the restaurants, Select the restaurants]
        
        <Example 10>
        Task:
        Show me the cafes nearby Yangjae station.
        Macro Action Plan:
        [Search for Yangjae station, Select Yangjae station, Search for cafes, Show the cafes, Select the cafes]
        ---
        
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
            
# class Actor():
#     def act(self, model, processor, macro_action_plan, current_macro_action, screenshot_path, previous_micro_action=None):
        
#         user_query = f"The user query: {current_macro_action}. The previous action : {previous_micro_action}."

#         dummy_image = Image.open(screenshot_path)
#         resized_height, resized_width  = smart_resize(dummy_image.height,
#             dummy_image.width,
#             factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
#             min_pixels=processor.image_processor.min_pixels,
#             max_pixels=processor.image_processor.max_pixels,)
#         mobile_use = MobileUse(
#             cfg={"display_width_px": resized_width, "display_height_px": resized_height}
#         )
        
#         prompt = NousFnCallPrompt()
#         raw_messages = [
#                 Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
#                 Message(role="user", content=[
#                     ContentItem(text=user_query),
#                     ContentItem(image=f"file://{screenshot_path}")
#                 ]),
#             ]

#         message_objs = prompt.preprocess_fncall_messages(
#             messages=raw_messages,
#             functions=[mobile_use.function],
#             lang=None,
#         )
        
#         message = [msg.model_dump() for msg in message_objs]

#         text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
#         inputs = processor(text=[text], images=[dummy_image], padding=True, return_tensors="pt").to('cuda')


#         output_ids = model.generate(**inputs, max_new_tokens=2048)
#         generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
#         output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

#         # Qwen will perform action thought function call
#         action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        
#         return action

class Actor():

    def act(self, model, processor, macro_action_plan, current_macro_action, screenshot_path, previous_micro_action=None):
        
        USER_QUERY = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{current_macro_action}
""" 

        dummy_image = Image.open(screenshot_path)
        resized_height, resized_width  = smart_resize(dummy_image.height,
            dummy_image.width,
            factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
            min_pixels=processor.image_processor.min_pixels,
            max_pixels=processor.image_processor.max_pixels,)

        raw_messages = [
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(text=USER_QUERY.format(current_macro_action=current_macro_action)),
                    ContentItem(image=f"file://{screenshot_path}")
                ]),
            ]

        message_objs = raw_messages

        message = [msg.model_dump() for msg in message_objs]

        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[dummy_image], padding=True, return_tensors="pt").to('cuda')


        output_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        action_type = output_text.split('Action:')[-1].strip()
        if "click" in action_type:
            match = re.search(r"\((\d+),\s*(\d+)\)", action_type)
            x, y = map(int, match.groups())
            action = {
                "name": "mobile_use",
                "arguments": {
                    "action": "click",
                    "coordinate": [x, y]
                }
            }
            pass
        elif "type" in action_type:
            match = re.search(r"content='(.*?)'", action_type)
            content = match.group(1)
            action = {
                "name": "mobile_use",
                "arguments": {
                    "action": "type",
                    "content": content
                }
            }
        elif "scroll" in action_type:
            box_match = re.search(r"start_box='\((\d+),\s*(\d+)\)'", action_type)
            dir_match = re.search(r"direction='(.*?)'", action_type)
            x, y = map(int, box_match.groups())
            direction = dir_match.group(1)
            if direction == "down":
                coordinate2 = [x, y - 100]
            elif direction == "up":
                coordinate2 = [x, y + 100]
            elif direction == "right":
                coordinate2 = [x - 100, y]
            elif direction == "left":
                coordinate2 = [x + 100, y]
            action = {
                "name": "mobile_use",
                "arguments": {
                    "action": "swipe",
                    "coordinate": [x, y],
                    "coordinate2": coordinate2
                }
            }
        elif "finished" in action_type:
            action = {
                "name": "mobile_use",
                "arguments": {
                    "action": "terminate",
                    "coordinate": [-1, -1]
                }
            }

        return(action)


class Verifier():
    
    def __init__(self):
        self.prompt = """You are a strict mobile‑task verifier.

Input
------
• screenshot_before : the screen **right before** the agent’s last action  
• screenshot_after  : the screen **right after** the agent’s last action  
• current_task      : one high‑level instruction (macro action)

Definitions
-----------
• End‑State  : the screen that appears **after the required UI element has been ACTIVATED and its effect is visible**  
               – e.g. map rerendered with specific filter badge ON, search result page OPENED, item actually in CART  
• Intermediate‑State : any transient screen such as pop‑up menus, text typed in a field, or highlight/selection **before** confirmation

Your Goal
---------
Return **1** only if screenshot_after shows the End‑State.  
If screenshot_after still shows an Intermediate‑State or gives no clear evidence, return **0**.

Hidden Reasoning Steps
----------------------
1. Restate current_task in your own words.  
2. Write explicit End‑State criteria for this task (what must be on‑screen).  
3. List Intermediate‑States that would NOT qualify.  
4. Compare screenshots and decide if End‑State is met.  
5. If any doubt or only Intermediate‑State is visible, decide “NOT completed”.

Output
------
Respond **ONLY** with this JSON (no extra text):

```json
{{
    "task_completed": 0 or 1,
    "reason": "<one concise sentence explaining the key visual evidence>"
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
