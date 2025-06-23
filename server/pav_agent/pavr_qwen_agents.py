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
        self.google_prompt =  """You are a helpful mobile agent and a good planner.
        You are given a screenshot of a mobile device and a task.
        You need to generate a macro action plan to complete the task.
        
        Below are some examples of tasks and their corresponding macro action plans.
        ---
        <Example 1>
        Task:
        Please display the route to Jejujib.
        Macro Action Plan:
        [Search for Jejujib, Show the routes]
        
        <Example 2>
        Task:
        Please display the route to Yori.
        Macro Action Plan:
        [Search for Yori, Show the routes]
        
        <Example 3>
        Task:
        Show me a wheelchair-accessible route to Seoul Forest Park.
        Macro Action Plan:
        [Search for Seoul Forest Park, Show the routes, Filter by wheelchair-accessible route, Show the filtered route]
        
        <Example 4>
        Task:
        Show me a less walking route to Gangnam station.
        Macro Action Plan:
        [Search for Gangnam station, Show the routes, Filter by less walking route, Show the filtered route]
        
        <Example 5>
        Task:
        Show me a fewer transfers route to Lotte Tower.
        Macro Action Plan:
        [Search for Lotte Tower, Show the routes, Filter by fewer transfers route, Show the filtered route]
        
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
        [Search for KAIST College of Business, Search for restaurants, Show the restaurants]
        
        <Example 10>
        Task:
        Show me the cafes nearby Yangjae station.
        Macro Action Plan:
        [Search for Yangjae station, Search for cafes, Show the cafes]
        ---
        
        Now you are given a task and a screenshot.
        Generae a macro action plan to complete the task.
        
        Task: 
        {instruction}
        Macro Aciton Plan:
        """
        # self.google_prompt =  """You are a helpful mobile agent and a good planner.
        # You are given a screenshot of a mobile device and a task.
        # You need to generate a macro action plan to complete the task.
        
        # Below are some examples of tasks and their corresponding macro action plans.
        # ---
        # <Example 1>
        # Task:
        # Please display the route to Jejujib.
        # Macro Action Plan:
        # [Search for Jejujib, Show the routes, Display the directions]
        
        # <Example 2>
        # Task:
        # Please display the route to Yori.
        # Macro Action Plan:
        # [Search for Yori, Show the routes, Display the directions]
        
        # <Example 3>
        # Task:
        # Show me a wheelchair-accessible route to Seoul Forest Park.
        # Macro Action Plan:
        # [Search for Seoul Forest Park, Show the routes, Display the directions, Filter by wheelchair-accessible route, Show the filtered route]
        
        # <Example 4>
        # Task:
        # Show me a less walking route to Gangnam station.
        # Macro Action Plan:
        # [Search for Gangnam station, Show the routes, Display the directions, Filter by less walking route, Show the filtered route]
        
        # <Example 5>
        # Task:
        # Show me a fewer transfers route to Lotte Tower.
        # Macro Action Plan:
        # [Search for Lotte Tower, Show the routes, Display the directions, Filter by fewer transfers route, Show the filtered route]
        
        # <Example 6>
        # Task:
        # Please add restaurant Gimgane to Favorites list.
        # Macro Action Plan:
        # [Search for restaurant Gimgane, Add Gimgane to Favorites list]
        
        # <Example 7>
        # Task:
        # Please add cafe Simjae to Want to go list.
        # Macro Action Plan:
        # [Search for cafe Simjae, Add Simjae to Want to go list]
        
        # <Example 8>
        # Task:
        # Show me the current rating of restaurant Sushiyoung.
        # Macro Action Plan:
        # [Search for restaurant Sushiyoung, Select restaurant Sushiyoung, Check the rating section]
        
        # <Example 9>
        # Task:
        # Show me the restaurants nearby KAIST College of Business.
        # Macro Action Plan:
        # [Search for KAIST College of Business, Select KAIST College of Business, Search for restaurants, Show the restaurants, Select the restaurants]
        
        # <Example 10>
        # Task:
        # Show me the cafes nearby Yangjae station.
        # Macro Action Plan:
        # [Search for Yangjae station, Select Yangjae station, Search for cafes, Show the cafes, Select the cafes]
        # ---
        
        # Now you are given a task and a screenshot.
        # Generae a macro action plan to complete the task.
        
        # Task: 
        # {instruction}
        # Macro Aciton Plan:
        # """
        
        self.ali_prompt =  """You are a helpful mobile agent and a good planner.
        You are given a screenshot of a mobile device and a task.
        You need to generate a macro action plan to complete the task.
        
        Below are some examples of tasks and their corresponding macro action plans.
        ---
        <Example 1>
        Task:
        Add Logitech MX Master 3S to cart.
        Macro Action Plan:
        [Search for Logitech MX Master 3S, Select Logitech MX Master 3S, Add to cart]
        
        <Example 2>
        Task:
        Add the most popular ramen to cart.
        Macro Action Plan:
        [Search for ramen, Filter by popularity, Select the first item, Add to cart]
        
        <Example 3>
        Task:
        Add the cheapest sofa to cart.
        Macro Action Plan:
        [Search for sofa, Filter by price, Select the first item, Add to cart]
        
        <Example 4>
        Task:
        Show the shoes on sale.
        Macro Action Plan:
        [Search for shoes, Filter by savings]
        
        <Example 5>
        Task:
        Show the items to ship in my orders.
        Macro Action Plan:
        [Navigate to my orders in account, Select to ship section, Show the items I need to pay]
        
        <Example 6>
        Task:
        Remove the third item in my cart.
        Macro Action Plan:
        [Navigate to cart tab, Select button next to third item, Remove the item]
        
        <Example 7>
        Task:
        View the items in rankings of K-VENUE page.
        Macro Action Plan:
        [Naviagate to K-VENUE page, Select rankings section, Show the items]
        
        <Example 8>
        Task:
        Add Shin ramyun to my wishlist
        Macro Action Plan:
        [Search for Shin ramyun, Select Shin ramyun, Select heart wish button]
        
        <Example 9>
        Task:
        Empty items in cart.
        Macro Action plan:
        [Navigate to cart tab, Select all, Remove the items]
        
        <Example 10>
        Task:
        Show the coupons I received.
        Macro Action Plan:
        [Navigate to coupons in account, Show coupons]
        ---
        
        Now you are given a task and a screenshot.
        Generae a macro action plan to complete the task.
        
        Task: 
        {instruction}
        Macro Aciton Plan:
        """
        self.composer_prompt =  """You are a helpful mobile agent and a good planner.
            You are given a screenshot of a mobile device and a task.
            You need to generate a macro action plan to complete the task.
            
            Below are some examples of tasks and their corresponding macro action plans.
            ---
            {few_shots}
            ---
            
            Now you are given a task and a screenshot.
            Generae a macro action plan to complete the task.
            
            Task: 
            {instruction}
            Macro Aciton Plan:
            """
            
        self.replan_success_prompt = """You are a helpful mobile agent and a good planner.
        You are given a screenshot of a mobile device, a task, a performed macro action, and a original macro action plan.
        You need to generate a new macro action plan to complete the task.
        
        Given screenshot is the screen **after** the performed macro action.
        And you planned to sequentially execute the original macro action plan in order to complete the given task.
        Now, based on the screenshot and the task, reassess whether your original macro action plan is still valid.
        If not, revise your plan and provide a newly updated macro action plan.
        
        Below are some examples of tasks and their corresponing macro action plans, replanned macro actions:
        ---
        <Example 1>
        Task:
        Add Logitech MX Master 3S to cart.
        Perfomed Macro Action:
        [Search for Logitech MX Master 3S, Select Logitech MX Master 3S]
        Original Macro Action Plan:
        [Buy the item]
        Updated Macro Action Plan:
        [Add to cart]
        
        <Example 2>
        Task:
        Add the most popular ramen to cart.
        Performed Macro Action:
        [Search for ramen]
        Original Macro Action Plan:
        [Filter by sale, Select the first item, Add to cart]
        Updated Macro Action Plan:
        [Filter by popularity, Select the first item, Add to cart]
        
        <Example 3>
        Task:
        Add the cheapest sofa to cart.
        Performed Macro Action:
        [Search for sofa, Filter by popularity]
        Original Macro Action Plan:
        [Select the first item, Add to cart]
        Updated Macro Action Plan:
        [Select the first item, Add to cart]
        ---
        
        Now you are given a task, a performed macro action, and an original macro action plan.
        Generate a newly updated macro action plan to complete the task using the same template as the original macro action plan.
        You can change the order of the macro actions, add new macro actions, remove existing macro actions, or paraphrase the macro actions
        And the returned macro action plan should consist only of the remaining steps to be executed, excluding any macro actions that have already been performed.
        **Never include any macro actions that have already been performed.**
        
        Task: 
        {instruction}
        Performed Macro Action: 
        {previous_macro_action}
        Original Macro Action Plan: 
        {macro_action_list}
        Updated Macro Action Plan:
        """
        
        self.replan_fail_prompt = """You are a helpful mobile agent and a good planner.
        You are given a history of screenshots of a mobile device, a task, a previous macro action, and a original macro action plan.
        You need to generate a new macro action plan to complete the task.
        
        Given a history of screenshot is the collection of screen to achieve a previous macro action, but you failied to achieve the previous macro action.
        So you need to think about why the previous macro action failed by examining the history of screenshots.
        And a original macro action plan is the sequence of plan you planned before to complete the given task.
        
        Now, based on the history of screenshots, the task, and the original macro action plan, you have to replan the macro action plan.
        You can change the order of the macro actions, add new macro actions, remove existing macro actions, or paraphrase the macro actions.
        
        Below are some examples of tasks and their corresponing macro action plans, replanned macro actions:
        ---
        <Example 1>
        Task:
        Add Logitech MX Master 3S to cart.
        Previous Macro Action:
        Select Logitech MX Master 3S
        Original Macro Action Plan:
        [Buy the item]
        Updated Macro Action Plan:
        [Choose Logitech MX Master 3S, Add to cart]
        
        <Example 2>
        Task:
        Add the most popular ramen to cart.
        Previous Macro Action:
        Search for ramen
        Original Macro Action Plan:
        [Filter by popularity, Select the first item, Add to cart]
        Updated Macro Action Plan:
        [Search for ramen, Filter by popularity, Select the first item, Add to cart]
        
        <Example 3>
        Task:
        Add the cheapest sofa to cart.
        Previous Macro Action:
        Filter by popularity
        Original Macro Action Plan:
        [Select the first item, Add to cart]
        Updated Macro Action Plan:
        [Filter by price, Select the first item, Add to cart]
        ---
        
        Now you are given a task, a previous macro action, and an original macro action plan.
        Generate a newly updated macro action plan to complete the task using the same template as the original macro action plan.
        You can change the order of the macro actions, add new macro actions, remove existing macro actions.
        I think you can also paraphrase the previous macro action to make it more suitable for the task.
        And the returned macro action plan should consist only of the remaining steps to be executed.
        
        Task: 
        {instruction}
        Previous Macro Action: 
        {previous_macro_action}
        Original Macro Action Plan: 
        {macro_action_list}
        Updated Macro Action Plan:
        """


        
    def plan(self, model, processor, task, screenshot_path, app_name, replan = "plan-initial", few_shots=None, macro_action_list=None, previous_macro_action=None, step=None, threshold=None):
        
       
        if replan == "plan-initial":          
            if few_shots is not None:
                prompt = self.composer_prompt.format(few_shots=few_shots, instruction=task)
                
            else:      
                if app_name == "google_maps":
                    prompt = self.google_prompt
                elif app_name == "ali":
                    prompt = self.ali_prompt
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": screenshot_path},
                        {"type": "text", "text": prompt.format(instruction=task)},
                    ],
                }
            ]
        
        elif replan == "replan-success":
            prompt = self.replan_success_prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": screenshot_path},
                        {"type": "text", "text": prompt.format(instruction=task, previous_macro_action=previous_macro_action, macro_action_list=macro_action_list)},
                    ],
                }
            ]

        elif replan == "replan-fail":
            prompt = self.replan_fail_prompt
            img_screenshots =  [{"type": "image", "image": f"./pavr_data/screenshot_{step-threshold+1+i}.png"} for i in range(threshold)]
            messages = [
                {
                    "role": "user",
                    "content": img_screenshots +[
                        {"type": "text", "text": prompt.format(instruction=task, previous_macro_action=previous_macro_action, macro_action_list=macro_action_list)},
                    ],
                }
            ]

        else:
            raise NotImplementedError
            
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
    def act(self, model, processor, macro_action_plan, current_macro_action, screenshot_path, previous_micro_action=None):
        
        user_query = f"The user query: {current_macro_action}. The previous action : {previous_micro_action}."

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

        self.prompt = """You are a strict mobile‑task verifier.
Input
------
• current_macro_action      : {macro_action}. one high‑level instruction (macro action)
• screenshot_before : the screen **right before** the agent’s last action
• screenshot_after  : the screen **right after** the agent’s last action

Definitions
-----------
• End‑State  : the screen that appears **after current task is completed**
                - e.g., "Search for <Something>" is completed when the search results <Something> are displayed
                - e.g., "Filter by <Something>" is completed when the filter badge <Something> is ON
• Intermediate‑State : any intermediate screen required to perform the task before it is completed.
                - e.g., any empty search bar, pop‑up menus, text typed in a field, or highlight/selection **before** confirmation


Hidden Reasoning Steps
----------------------
1. Restate current_macro_action in your own words.
2. Think about Expectation of End‑State criteria for this task (what must be on‑screen).
3. Think about Intermediate‑States that would NOT qualify.
4. Compare screenshots and decide if End‑State is met.
5. If any doubt or only Intermediate‑State is visible, decide “NOT completed”.

Your Goal
---------
Your current task is "{macro_action}".
Return **1** only if screenshot_after meets the expectation of the task completion.
If screenshot_after still shows an Intermediate‑State or gives no clear evidence, return **0**.

Output
------
```json
{{
    "current_macro_action": <current_macro_action {macro_action}>,
    "expectation": <one concise sentence describing after the task completion>,
    "comparison": <one concise sentence describing the screeshot_after meets the expectation>,
    "task_completed": 0 or 1
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
