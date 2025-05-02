import json
import re
import ast
from PIL import Image, ImageDraw, ImageFont, ImageColor
from IPython.display import display

from qwen_vl_utils import smart_resize
from pav_agent.pav_nous_fncall_prompt import NousFnCallPrompt, Message, ContentItem
from pav_agent.pav_agent_function_call import PlannerUse, ActorUse

def agent_preprocess(processor, screenshot):
    # The resolution of the device will be written into the system prompt. 
    dummy_image = Image.open(screenshot)
    resized_height, resized_width  = smart_resize(dummy_image.height,
        dummy_image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,)
    return dummy_image, resized_height, resized_width

class Planner():
    def planner_prompt(self):
        prompt = f"""
Use a touchscreen to interact with a mobile device, and take screenshots.

* This is an interface to a mobile device with touchscreen. You are a Planner Agent and you to plan the macro actions in order to achieve the given instruction.

* Instruction is the main goal to achieve. The instruction for you is {self.user_query}
* Macro actions are long-term actions such as 'Open the browser.', 'Search cafes nearby Samsung Seoul R&D Campus', 'Navigate to the bookstore' etc.
* Micro actions are short-term actions like clicking and typing that are directly performed on the device.

* The macro action sequence plays a crucial role in guiding the actor toward executing more precise micro actions to accomplish the final instruction.
* The planned future macro action sequence should be provided to the user at the end.

* The planned macro actions for previous steps are {self.prev_macro_action_plan_str}.
* You have to first determine your current status and than plan the next macro actions till the end of the 
* The screen's resolution is {self.resized_width}x{self.resized_height}.

* Never output micro actions!!
* Make sure to provide specific names of any applications, items, icons, locations, options. Make sure to provide macro action 'Done_Actions' at the end of action sequence if you are done planning.

Plan the next macro action sequence to achieve the instruction using the below tools.

# Tools
You are provided with function signatures within <tools></tools> XML tags:
<tools>
* macro_action description : [
    The macro action to perform. The available macro actions are:
    * `open`: Open an app on the device.
    * `install` : Install an app on the device.
    * `search` : Search and item or location, menu the user is interested in.
    * `select` : Choose an option, item, location, etc.
    * `set_A_to_B` : Assign a specific target value or status B to a setting or parameter A.
    * `change_A_to_B` : Modify the current value or state of A into a target value or state B.
    * `navigate_to` : Display the map route from the current location to the specified target destination.
    * `add_A_to_B` : Add a target item A into list B.
    * `filter_route` : Apply specific conditions to limit and display only matching navigation routes.
    * `show` : Show a certain list.
    * `terminate`: Terminate the current task and report its completion status.
    ]
* macro_action enum : [
    "open",
    "install",
    "search",
    "select"
    "set_A_to_B",
    "change_A_to_B",
    "navigate_to",
    "add_A_to_B",
    "filter_route",
    "show"
    "terminate",
    ],
* macro_action type : string
* macro_action parameters : [
    "app": [
        "description": "Provide the name of the necessary application. Required only by 'macro_action=open' and 'macro_action=install'.",
        "type": "string",
    ],
    "target": [
        "description": "Provide the specific name of the location or item. Required only by 'macro_action=search', 'macro_action=select', 'macro_action=add', and 'macro_action=navigate_to'.",
        "type": "string",
    ],
    "list": [
        "description": "'Cart' means a list of selected items to purchase, 'Menu' means a list of available options or functions that the user can select, 'Want to go list' means a list of places the user intends to visit in the future functioning like a wishlist for locations, and 'Favorite list' means a personalized collection of frequently accessed or preferred items by the user. Required only by 'macro_action=add' and 'macro_action=show'.",
        "enum": [
            "Cart",
            "Menu",
            "Want to go list"
            "Favorite list"
        ],
        "type": "string",
    ],
    "button": [
        "description": "'Save' means a button used to store or retain selected items, 'Cart' means an icon representing a shopping cart which is used to view or access items selected for purchase, 'Menu' means a button that opens a list of navigation or action options, 'Direction' means a button for navigation guidance that helps the user reach a destination, and 'Done' means a button to finally add an element to a list. Required only by 'macro_action=select'.",
        "enum": [
            "Save",
            "Cart",
            "Menu",
            "Direction"
        ],
        "type": "string",
    ],
    "food_set_option": [
        "description": "'A la carte' means ordering individual dishes separately from the menu, 'Extra value meal' means a meal combo that includes a main item, side, and drink at a discounted price, and 'Large extra value meal' means a meal combo in large size that includes a main item, side, and drink at a discounted price. Required only by 'macro_action=select'.",
        "enum": [
            "A la carte",
            "Extra value meal",
            "Large extra value meal",
        ],
        "type": "string",
    ],
    "food_side_option": [
        "description": "'Side menu' means a sliding panel or secondary menu such as fries, 'Drink' means a beverage item such as cola or coffee, and 'Dessert' means a sweet dish usually served at the end of a meal such as pie. Required only by 'macro_action=select', 'macro_action=set', and 'macro_action=change'.",
        "enum": [
            "Side menu",
            "Drink",
            "Dessert",
        ],
        "type": "string",
    ],
    "route_option": [
        "description": "'Nearby' means located within a short distance, 'Wheelchair-accessible route' means a route that can be used by people in wheelchairs, 'Less walking route' means a route that requires less walking distance, and 'Fewer transfers route' means a route that involves fewer transfers between lines or vehicles. Required only by 'macro_action=show_route'.",
        "enum": [
            "Nearby",
            "Wheelchair-accessible route",
            "Less walking route",
            "Fewer transfers route",
        ],
        "type": "string",
    ],
    "status": [
        "description": "The status of the task. Required only by 'macro_action=terminate'.",
        "type": "string",
        "enum": ["Done_Actions", "Replan_Actions"],
    ]
]
</tools>

For each function call, return json objects with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
{{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
{{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
{{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
{{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
{{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
{{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
</tool_call>"""
        prompt2 = f"""
Use a touchscreen to interact with a mobile device, and take screenshots.

* This is an interface to a mobile device with touchscreen. You are a Planner Agent and you to plan the macro actions in order to achieve the given instruction.

* Instruction is the main goal to achieve. The instruction for you is {self.user_query}
* Macro actions are long-term actions such as 'Open the browser.', 'Search cafes nearby Samsung Seoul R&D Campus', 'Navigate to the bookstore' etc.
* Micro actions are short-term actions like clicking and typing that are directly performed on the device.

* The macro action sequence plays a crucial role in guiding the actor toward executing more precise micro actions to accomplish the final instruction.
* The planned future macro action sequence should be provided to the user at the end.

* The planned macro actions for previous steps are {self.prev_macro_action_plan_str}.
* You have to first determine your current status and than plan the next macro actions till the end of the 
* The screen's resolution is {self.resized_width}x{self.resized_height}.

* Never output micro actions!!
* Make sure to provide specific names of any applications, items, icons, locations, options. Make sure to provide macro action 'Done_Actions' at the end of action sequence if you are done planning.

Plan the next macro actions using the below tools.

# Tools
You are provided with function signatures within <tools></tools> XML tags:
<tools>
* macro_action description : [
    The macro action to perform. The available macro actions are:
    * `open`: Open an app on the device.
    * `install` : Install an app on the device.
    * `search` : Search and item or location, menu the user is interested in.
    * `select` : Choose an option, item, location, etc.
    * `set_A_to_B` : Assign a specific target value or status B to a setting or parameter A.
    * `change_A_to_B` : Modify the current value or state of A into a target value or state B.
    * `navigate_to` : Display the map route from the current location to the specified target destination.
    * `add_A_to_B` : Add a target item A into list B.
    * `filter_route` : Apply specific conditions to limit and display only matching navigation routes.
    * `show` : Show a certain list.
    * `terminate`: Terminate the current task and report its completion status.
    When you want to add a location to a list, you should usually search the location, select the save button, select the list you want to save the location to, and than select the Done button.
    When you want to add or change a side menu, you should usually select the change button.
    ]

* macro_action enum : [
    "open",
    "install",
    "search",
    "select"
    "set_A_to_B",
    "change_A_to_B",
    "navigate_to",
    "add_A_to_B",
    "filter_route",
    "show"
    "terminate",
    ],

* macro_action type : string

* macro_action parameters : [
    "app": [
        "description": "Provide the name of the necessary application. Required only by 'macro_action=open' and 'macro_action=install'.",
        "type": "string",
    ],
    "target": [
        "description": "Provide the specific name of the location or item. Required only by 'macro_action=search', 'macro_action=select', 'macro_action=add', and 'macro_action=navigate_to'.",
        "type": "string",
    ],
    "list": [
        "description": "'Cart' means a list of selected items to purchase, 'Menu' means a list of available options or functions that the user can select, 'Want to go list' means a list of places the user intends to visit in the future functioning like a wishlist for locations, and 'Favorite list' means a personalized collection of frequently accessed or preferred items by the user. Required only by 'macro_action=add_A_to_B' and 'macro_action=show'.",
        "enum": [
            "Cart",
            "Menu",
            "Want to go list"
            "Favorite list"
        ],
        "type": "string",
    ],
    "button": [
        "description": "'Save' means a button used to store or retain selected items, 'Cart' means an icon representing a shopping cart which is used to view or access items selected for purchase, 'Menu' means a button that opens a list of navigation or action options, 'Direction' means a button for navigation guidance that helps the user reach a destination, and 'Done' means a button to save the list after adding an element to the list. Required only by 'macro_action=select'.",
        "enum": [
            "Save",
            "Cart",
            "Menu",
            "Direction",
            "Done"
        ],
        "type": "string",
    ],
    "food_set_option": [
        "description": "'A la carte' means ordering individual dishes separately from the menu, 'Extra value meal' means a meal combo that includes a main item, side, and drink at a discounted price, and 'Large extra value meal' means a meal combo in large size that includes a main item, side, and drink at a discounted price. Required only by 'macro_action=select'.",
        "enum": [
            "A la carte",
            "Extra value meal",
            "Large extra value meal",
        ],
        "type": "string",
    ],
    "food_side_option": [
        "description": "'Side menu' means a sliding panel or secondary menu such as fries, 'Drink' means a beverage item such as cola or coffee, and 'Dessert' means a sweet dish usually served at the end of a meal such as pie. Required only by 'macro_action=select', 'macro_action=set_A_to_B', and 'macro_action=change_A_to_B'.",
        "enum": [
            "Side menu",
            "Drink",
            "Dessert",
        ],
        "type": "string",
    ],
    "route_option": [
        "description": "'Nearby' means located within a short distance, 'Wheelchair-accessible route' means a route that can be used by people in wheelchairs, 'Less walking route' means a route that requires less walking distance, and 'Fewer transfers route' means a route that involves fewer transfers between lines or vehicles. Required only by 'macro_action=show_route'.",
        "enum": [
            "Nearby",
            "Wheelchair-accessible route",
            "Less walking route",
            "Fewer transfers route",
        ],
        "type": "string",
    ],
    "status": [
        "description": "The status of the task. Required only by 'macro_action=terminate'.",
        "type": "string",
        "enum": ["Done_Actions", "Replan_Actions"],
    ]
    ]
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
    {{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
    {{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
    {{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
    {{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
    {{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
    {{"macro-actions": <macro-actions>, "arguments": <args-json-object>}}
    </tool_call>"""#.strip()
        
        return prompt2

    def planner(self, model, processor, screenshot, user_query, prev_macro_action_plan):
        self.user_query = user_query
        self.dummy_image, self.resized_height, self.resized_width = agent_preprocess(processor, screenshot)
        self.prev_macro_action_plan_str = "\n".join(prev_macro_action_plan)

        # Build Messages

        # Method 1 : use 'PlannerUse'
        # planner = PlannerUse(cfg={"display_width_px": self.resized_width, "display_height_px": self.resized_height, "prev_macro_action_plan": prev_macro_action_plan})
        # message = NousFnCallPrompt().preprocess_fncall_messages(
        #     messages = [ 
        #         Message(role="planner", content=[
        #             ContentItem(text=user_query),
        #             ContentItem(image=f"file://{screenshot}")
        #         ]),
        #     ],
        #     functions=[planner.function],
        #     lang=None,
        # )
        # message = [msg.model_dump() for msg in message]
        
        # Method 2 : prompt tuning
        prompt = self.planner_prompt()
        messages = [
                Message(role="user", content=[
                    ContentItem(image=f"file://{screenshot}"),
                    ContentItem(text=prompt),
                    ]
                )
            ]
        message = [msg.model_dump() for msg in messages]
        # print("message : ", message)

        # Preparation for Inference
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        # print("text : ",text)
        inputs = processor(text=[text], images=[self.dummy_image], padding=True, return_tensors="pt").to('cuda')

        # Inference
        output_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        print('planner output : ', output_text)

        # Extract Macro Actions
        macro_action_json_blocks = re.findall(r"```json\s*(.*?)\s*```", output_text, re.DOTALL)
        macro_actions = []
        for macro_action in macro_action_json_blocks:
            print(macro_action)
            try:
                macro_actions.append(json.loads(macro_action))
            except json.JSONDecodeError as e:
                print(f"Error: {e}")
        
        # Refine Macro Actions to input Actor
        macro_actions = self.to_actor(macro_actions)
        return macro_actions
    
    def to_actor(self, macro_actions):
        # Refine Macro Actions : {'macro-actions': 'add_A_to_B', 'arguments': {'A': 'RestaurantYori', 'B': 'Wanttogolist'}} -> 'add Restaurant Yori to Want to go list'
        refined_macro_action_plan = []
        for action in macro_actions:
            action_str = json.dumps(action)
            # print("action : ", action)
            parsed_action = ast.literal_eval(action_str)

            refined_macro_action = parsed_action.get("macro-actions")
            refined_argument = parsed_action.get("arguments")
            if refined_argument.get("app") != None:
                refined_macro_action_plan.append(refined_macro_action + " " + refined_argument.get("app"))
            if refined_argument.get("target") != None:
                refined_macro_action_plan.append(refined_macro_action + " " + refined_argument.get("target"))
            # if refined_argument.get("list") != None:########### 수정 필요
            #     if refined_macro_action == 'add':
            #         refined_macro_action_plan.append(refined_macro_action + " to " + refined_argument.get("list"))
            #     else:
            #         refined_macro_action_plan.append(refined_macro_action + " " + refined_argument.get("list"))
            if refined_argument.get("button") != None:
                refined_macro_action_plan.append(refined_macro_action + " " + refined_argument.get("button"))
            if refined_argument.get("food_set_option") != None:
                refined_macro_action_plan.append(refined_macro_action + " " + refined_argument.get("food_set_option"))
            if refined_argument.get("food_side_option") != None:########### 수정 필요
                refined_macro_action_plan.append(refined_macro_action + " " + refined_argument.get("food_side_option"))
            if refined_argument.get("route_option") != None:
                refined_macro_action_plan.append(refined_macro_action + " " + refined_argument.get("route_option"))
            if (refined_argument.get("A") and refined_argument.get("B")) != None:
                A = refined_argument.get("A")
                B = refined_argument.get("B")
                if refined_macro_action == 'set_A_to_B':
                    macro = 'set'
                elif refined_macro_action == 'change_A_to_B':
                    macro = 'change'
                else:
                    macro = 'add'
                refined_macro_action_plan.append(macro + " " + A + " to " + B)
            if refined_argument.get("status") != None:
                refined_macro_action_plan.append(refined_argument.get("status"))
        return refined_macro_action_plan

            
class Actor():
    def actor(self, model, processor, screenshot, curr_macro_action):
        self.dummy_image, self.resized_height, self.resized_width = agent_preprocess(processor, screenshot)
        self.curr_macro_action = "\n".join(curr_macro_action)
        actor = ActorUse(cfg={"display_width_px": self.resized_width, "display_height_px": self.resized_height})

        # Build messages
        message = NousFnCallPrompt().preprocess_fncall_messages(
            messages = [
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(text=curr_macro_action),
                    ContentItem(image=f"file://{screenshot}")
                    ]
                )
            ],
            functions=[actor.function],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]
        # print("message : ", message)

        # Preparation for inference
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        # print("text : ",text)
        inputs = processor(text=[text], images=[self.dummy_image], padding=True, return_tensors="pt").to('cuda')

        # Inference
        output_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        print('output : ', output_text)

        # Extract Micro Actions
        action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        self.visualize_click(action)
        return action
    
    def to_emulator(action):
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
        
    def draw_point(self, image: Image.Image, point: list, color=None):
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
    
    def visualize_click(self, action):
        # Visualize Micro Action "click" : Draw a green circle onto the image.
        display_image = self.dummy_image.resize((self.resized_width, self.resized_height))
        if action['arguments']['action'] == "click":
            display_image = self.draw_point(self.dummy_image, action['arguments']['coordinate'], color='green')
            display(display_image)
        else:
            display(display_image)


class Verifier():
    def verifier_prompt(self):
        prompt = f"""
Determine whether the action has been completed by examining the following two screenshots.
If the action has not been completed yet, return 0. If the action has been completed, return 1.
Action: {self.macro_action}

Think step by step and provide the final answer. And return the answer in the following format:
<verify>
{{
    "action_completed": 0,
    "reason": "The action has not been completed yet."
}}
</verify>
"""
        return prompt
        
    def verifier(self, model, processor, screenshot1, screenshot2, macro_action):
        self.dummy_image1, self.resized_height1, self.resized_width1 = agent_preprocess(processor, screenshot1)
        self.dummy_image2, self.resized_height2, self.resized_width2 = agent_preprocess(processor, screenshot2)
        self.macro_action = macro_action
        
        # Build Messages
        prompt = self.verifier_prompt()
        message = [
            Message(role="system", content=[ContentItem(text="You are a helpful mobile agent and a good verifier")]),
            Message(role="user", content=[
                ContentItem(text=prompt),
                ContentItem(image=f"file://{screenshot1}"),
                ContentItem(image=f"file://{screenshot2}")
                ]
            )
        ]
        message = [msg.model_dump() for msg in message]
        # print("message : ", message)

        # Preparation for Inference
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        # print("text",text)
        inputs = processor(text=[text], images=[self.dummy_image1, self.dummy_image2], padding=True, return_tensors="pt").to('cuda')

        # Inference
        output_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        print('verifier output : ', output_text)

        # Extract Verification Information
        verification_info = re.findall(r"```json\s*(.*?)\s*```", output_text, re.DOTALL)
        print(verification_info)
        verification_info_str = ast.literal_eval(verification_info[0])
        verification = verification_info_str.get("action_completed")
        reason = verification_info_str.get("reason")
        print("verify : ", verification)
        print("reason : ", reason)
        return int(verification)
