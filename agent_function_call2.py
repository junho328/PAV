from typing import Union, Tuple, List

from qwen_agent.tools.base import BaseTool, register_tool

print("WAYIRANO")

@register_tool("mobile_use2")
class MobileUse2(BaseTool):
    @property
    def description(self):
        return f"""
Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "click",
                    "long_press",
                    "swipe",
                    "type",
                    "system_button",
                    "open",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
                "type": "array",
            },
            "coordinate2": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=key`, `action=type`, and `action=open`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                "type": "number",
            },
            "button": {
                "description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`",
                "enum": [
                    "Back",
                    "Home",
                    "Menu",
                    "Enter",
                ],
                "type": "string",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action == "key":
            return self._key(params["text"])
        elif action == "click":
            return self._click(
                coordinate=params["coordinate"]
            )
        elif action == "long_press":
            return self._long_press(
                coordinate=params["coordinate"], time=params["time"]
            )
        elif action == "swipe":
            return self._swipe(
                coordinate=params["coordinate"], coordinate2=params["coordinate2"]
            )
        elif action == "type":
            return self._type(params["text"])
        elif action == "system_button":
            return self._system_button(params["button"])
        elif action == "open":
            return self._open(params["text"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Unknown action: {action}")

    def _key(self, text: str):
        raise NotImplementedError()
        
    def _click(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _long_press(self, coordinate: Tuple[int, int], time: int):
        raise NotImplementedError()

    def _swipe(self, coordinate: Tuple[int, int], coordinate2: Tuple[int, int]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()

    def _system_button(self, button: str):
        raise NotImplementedError()

    def _open(self, text: str):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()
    


@register_tool("actor_use")
class ActorUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, scrolling, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `click`: Click the point on the screen with coordinate (x, y).
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `press`: Press the system button.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "click",
                    "swipe",
                    "type",
                    "press",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click` and `action=swipe`.",
                "type": "array",
            },
            "coordinate2": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=type`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                "type": "number",
            },
            "button": {
                "description": "Back means returning to the previous interface, Home means returning to the home screen, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=press`",
                "enum": [
                    "Back",
                    "Home",
                    "Menu",
                    "Enter",
                ],
                "type": "string",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action == "click":
            return self._click(
                coordinate=params["coordinate"]
            )
        elif action == "swipe":
            return self._swipe(
                coordinate=params["coordinate"], coordinate2=params["coordinate2"]
            )
        elif action == "type":
            return self._type(params["text"])
        elif action == "press":
            return self._system_button(params["button"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Unknown action: {action}")
        
    def _click(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _swipe(self, coordinate: Tuple[int, int], coordinate2: Tuple[int, int]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()

    def _press(self, button: str):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()
    


@register_tool("planner_use")
class PlannerUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a touchscreen to interact with a mobile device, and take screenshots.

* This is an interface to a mobile device with touchscreen. You are a Planner Agent and you to plan the macro actions in order to achieve the given instruction.

* Instruction is the main goal to achieve such as 'Show me the cafes nearby Samsung Seoul R&D Campus'.
* Macro actions are long-term actions such as 'Open the browser.', 'Search cafes nearby Samsung Seoul R&D Campus', 'Navigate to the bookstore' etc.
* Micro actions are short-term actions like clicking and typing that are directly performed on the device.

* The macro action sequence plays a crucial role in guiding the actor toward executing more precise micro actions to accomplish the final instruction.
* The planned future macro action sequence should be provided to the user at the end.

* The planned macro actions for previous steps are {self.prev_macro_action_plan}.
* You have to first determine your current status and than plan the next macro actions till the end of the 
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.

* Make sure to provide specific names of any applications, items, icons, locations, options. Make sure to provide 'Finish_Actions' at the end of the macro action sequence if you are done planning.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The macro action to perform. The available macro actions are:
* `open`: Open an app on the device.
* `install` : Install an app on the device.
* `search` : Search and item or location, menu the user is interested in.
* `select` : Choose an option, item, location, etc.
* `set_A_to_B` : Assign a specific target value or status B to a setting or parameter A.
* `change_A_to_B` : Modify the current value or state of A into a target value or state B.
* `navigate_to` : Display the map route from the current location to the specified target destination.
* `add_A_to_B` : Add a target item A into list B.
* `filter_route` : Filter the route.
* `show` : Show a certain list.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
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
                "type": "string",
            },
            "app": {
                "description": "Provide the name of the necessary application. Required only by 'action=open' and 'action=install'.",
                "type": "string",
            },
            "target": {
                "description": "Provide the specific name of the location or item. Required only by 'action=search', 'action=select', 'action=add', and 'action=navigate_to'.",
                "type": "string",
            },
            "list": {
                "description": "'Cart' means a list of selected items to purchase, 'Menu' means a list of available options or functions that the user can select, 'Want to go list' means a list of places the user intends to visit in the future functioning like a wishlist for locations, and 'Favorite list' means a personalized collection of frequently accessed or preferred items by the user. Required only by 'action=add' and 'action=show'.",
                "enum": [
                    "Cart",
                    "Menu",
                    "Want to go list"
                    "Favorite list"
                ],
                "type": "string",
            },
            "button": {
                "description": "'Save' means a button used to store or retain selected items, 'Cart' means an icon representing a shopping cart which is used to view or access items selected for purchase, 'Menu' means a button that opens a list of navigation or action options, and 'Direction' means a button for navigation guidance that helps the user reach a destination. Required only by 'action=select'.",
                "enum": [
                    "Save",
                    "Cart",
                    "Menu",
                    "Direction"
                ],
                "type": "string",
            },
            "food_set_option": {
                "description": "'A la carte' means ordering individual dishes separately from the menu, 'Extra value meal' means a meal combo that includes a main item, side, and drink at a discounted price, and 'Large extra value meal' means a meal combo in large size that includes a main item, side, and drink at a discounted price. Required only by 'action=select'.",
                "enum": [
                    "A la carte",
                    "Extra value meal",
                    "Large extra value meal",
                ],
                "type": "string",
            },
            "food_side_option": {
                "description": "'Side menu' means a sliding panel or secondary menu such as fries, 'Drink' means a beverage item such as cola or coffee, and 'Dessert' means a sweet dish usually served at the end of a meal such as pie. Required only by 'action=select', 'action=set', and 'action=change'.",
                "enum": [
                    "Side menu",
                    "Drink",
                    "Dessert",
                ],
                "type": "string",
            },
            "route_option": {
                "description": "'Nearby' means located within a short distance, 'Wheelchair-accessible route' means a route that can be used by people in wheelchairs, 'Less walking route' means a route that requires less walking distance, and 'Fewer transfers route' means a route that involves fewer transfers between lines or vehicles. Required only by 'action=show_route'.",
                "enum": [
                    "Nearby",
                    "Wheelchair-accessible route",
                    "Less walking route",
                    "Fewer transfers route",
                ],
                "type": "string",
            },
            "status": {
                "description": "The status of the task. Required only by 'action=terminate'.",
                "type": "string",
                "enum": ["Finish_Actions", "Replan_Actions"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        self.prev_macro_action_plan = cfg["prev_macro_action_plan"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action == "open":
            return self._open(params["text"])
        elif action == "install":
            return self._install(coordinate=params["coordinate"])
        elif action == "search":
            return self._search(params["target"])
        elif action == "select":
            return self._select(params["target"]) or self._select(params["button"]) or self._select(params["food_set_option"]) or self._select(params["food_side_option"])
        elif action == "set_A_to_B":
            return self._set_A_to_B(A=params["food_side_option"], B=params["target"])
        elif action == "change_A_to_B":
            return self._change_A_to_B(A=params["food_side_option"], B=params["target"])
        elif action == "navigate_to":
            return self._navigate_to(params["target"])
        elif action == "add_A_to_B":
            return self._add_A_to_B(A=params["target"], B=params["list"])
        elif action == "show_route":
            return self._show_route(params["route_options"])
        elif action == "show":
            return self._show(params["list"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Unknown action: {action}")

    def _open(self, app: str):
        raise NotImplementedError()
        
    def _install(self, app: str):
        raise NotImplementedError()

    def _search(self, target: Tuple[int, int], time: int):
        raise NotImplementedError()

    def _select(self, target: str):
        raise NotImplementedError()

    def _set_A_to_B(self, food_side_option: str, target: str):
        raise NotImplementedError()

    def _change_A_to_B(self, food_side_option: str, target: str):
        raise NotImplementedError()

    def _navigate_to(self, target: str):
        raise NotImplementedError()

    def _add_A_to_B(self, target: str, list: str):
        raise NotImplementedError()
    
    def _show_route(self, route_option: int):
        raise NotImplementedError()
    
    def _show(self, list: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()