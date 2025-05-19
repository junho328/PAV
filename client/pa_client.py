import base64, io, subprocess, tempfile, requests, json, time
import argparse
from pathlib import Path
from PIL import Image
import shlex

def take_screenshot(args, step: int) -> bytes:
    """
    ADB를 사용해 에뮬레이터 화면을 캡처하고 지정된 위치에 저장하며 PNG 바이트를 반환
    """
    image_path = args.image_path + '/'+ args.task_number
    output_file = Path(image_path) / f"screenshot_{step}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # adb로 임시 스크린샷 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        subprocess.run(
            ["adb", "-s", "emulator-5554", "exec-out", "screencap", "-p"],
            stdout=tmp_file,
            check=True
        )
        tmp_path = Path(tmp_file.name)

    # 이미지 읽고 저장
    with open(tmp_path, "rb") as f:
        image_bytes = f.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.save(output_file)

    tmp_path.unlink()
    return image_bytes

def send_to_server(args, task, image_bytes, step, role, previous_action, app_name) -> dict:
    """
    서버로 task + base64 이미지 전송 → 응답 JSON 반환
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"task": task, "image_base64": b64, "step": step, "role": role, "previous_action": previous_action, "app_name": app_name}

    r = requests.post(args.server, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def adb_shell(*args):
    subprocess.run(["adb", "-s", "emulator-5554", "shell"] + list(args), check=True)

def qwen_action(response: dict) -> str:
    
    response = response["arguments"]
    
    action_type = response.get("action", "")
    
    if action_type == "system_button":
        
        button = response.get("button")
        if not button:
            print("system_button requires [button]")
            return action_type
        
        key_map = {
            "HOME": "3",
            "BACK": "4",
            "MENU": "82",
            "ENTER": "66"
        }
        key_code = key_map.get(button.upper())
        if key_code:
            adb_shell("input", "keyevent", key_code)
        else:
            print(f"Unknown system button: {button}")

    elif action_type == "click" or action_type == "left_click":
        
        coordinate = response.get("coordinate")
        
        if len(coordinate) < 2:
            print("click requires [x, y]")
            return "click"
        
        x, y = int(coordinate[0]), int(coordinate[1])
        adb_shell("input", "tap", str(x), str(y))

    elif action_type == "swipe":
        
        coordinate = response.get("coordinate")
        
        if len(coordinate) < 2:
            print("swipe requires coordinate [x, y]")
            return action_type
        
        coordinate2 = response.get("coordinate2")
        
        if len(coordinate2) < 2:
            print("swipe requires coordinate2 [x, y]")
            return action_type
        
        x1, y1, x2, y2 = int(coordinate[0]), int(coordinate[1]), int(coordinate2[0]), int(coordinate2[1])
        adb_shell("input", "swipe", str(x1), str(y1), str(x2), str(y2))

    elif action_type == "type":
        
        text = response.get("text")
        if not text:
            print("type requires [text]")
            return action_type
        
        adb_shell("input", "text", shlex.quote(text))

    elif action_type == "long_click":
        
        duration = response.get("time")
        
        if not duration:
            print("long_click requires [time]")
            return action_type

        adb_shell("input", "swipe", str(x), str(y), str(x), str(y), str(duration))

    elif action_type == "key":
        
        key_num = response.get("text")
        if not key_num:
            print("key requires [text]")
            return action_type
        
        adb_shell("input", "keyevent", str(key_num))

    elif action_type == "open":
        
        package_name = response.get("text")
        if not package_name:
            print("open requires [text]")
            return action_type
        
        adb_shell("monkey", "-p", package_name, "-c", "android.intent.category.LAUNCHER", "1")

    elif action_type == "wait":
        
        seconds = response.get("time")
        
        if not seconds:
            print("wait requires [time]")
            return action_type
        
        seconds = int(seconds)
        time.sleep(seconds)

    elif action_type == "terminate":
        
        status = response.get("status")
        if not status:
            print("terminate requires [status]")
            return action_type
        
        print(f"Task terminated with status: {status}")
        
    else:
        print(f"Unknown action type: {action_type}")
        return action_type

    return action_type


def uitars_action(response: dict) -> str:
    
    response = response["arguments"]
    
    action_type = response.get("action", "")

    # if action_type == "click":
    #     coord = response.get("start_box")
    #     if coord and len(coord) == 2:
    #         x, y = map(int, coord)
    #         adb_shell("input", "tap", str(x), str(y))
    #     else:
    #         print("click requires 'start_box': [x, y]")
    #         return action_type
    
    if action_type == "click":
        coordinate = response.get("coordinate")
        
        if len(coordinate) < 2:
            print("click requires [x, y]")
            return "click"
        
        x, y = int(coordinate[0]), int(coordinate[1])
        adb_shell("input", "tap", str(x), str(y))

    # elif action_type == "long_press":
    #     coord = response.get("start_box")
    #     if coord and len(coord) == 2:
    #         x, y = map(int, coord)
    #         duration = 1000  # 1초간 long press
    #         adb_shell("input", "swipe", str(x), str(y), str(x), str(y), str(duration))
    #     else:
    #         print("long_press requires 'start_box': [x, y]")
    #         return action_type

    # elif action_type == "type":
    #     text = response.get("content")
    #     if text:
    #         escaped_text = shlex.quote(text)
    #         adb_shell("input", "text", escaped_text)
    #     else:
    #         print("type requires 'content'")
    #         return action_type

    elif action_type == "type":
        
        text = response.get("content")
        if not text:
            print("type requires [text]")
            return action_type
        
        adb_shell("input", "text", shlex.quote(text))

    elif action_type == "swipe":
        
        coordinate = response.get("coordinate")
        
        if len(coordinate) < 2:
            print("swipe requires coordinate [x, y]")
            return action_type
        
        coordinate2 = response.get("coordinate2")
        
        if len(coordinate2) < 2:
            print("swipe requires coordinate2 [x, y]")
            return action_type
        
        x1, y1, x2, y2 = int(coordinate[0]), int(coordinate[1]), int(coordinate2[0]), int(coordinate2[1])
        adb_shell("input", "swipe", str(x1), str(y1), str(x2), str(y2))

    elif action_type == "terminate":
        
        status = response.get("status")
        if not status:
            print("terminate requires [status]")
            return action_type
        
        print(f"Task terminated with status: {status}")

    # elif action_type == "scroll":
    #     coord = response.get("start_box")
    #     direction = response.get("direction")
    #     scroll_offset = 500  # 기본 스크롤 픽셀 이동량

    #     if coord and len(coord) == 2 and direction in {"up", "down", "left", "right"}:
    #         x1, y1 = map(int, coord)
    #         if direction == "up":
    #             x2, y2 = x1, y1 - scroll_offset
    #         elif direction == "down":
    #             x2, y2 = x1, y1 + scroll_offset
    #         elif direction == "left":
    #             x2, y2 = x1 - scroll_offset, y1
    #         else:  # right
    #             x2, y2 = x1 + scroll_offset, y1
    #         adb_shell("input", "swipe", str(x1), str(y1), str(x2), str(y2), "300")
    #     else:
    #         print("scroll requires 'start_box' and valid 'direction'")
    #         return action_type

    # elif action_type == "open_app":
    #     package_name = response.get("app_name")
    #     if package_name:
    #         adb_shell("monkey", "-p", package_name, "-c", "android.intent.category.LAUNCHER", "1")
    #     else:
    #         print("open_app requires 'app_name'")
    #         return action_type

    # elif action_type == "drag":
    #     start = response.get("start_box")
    #     end = response.get("end_box")
    #     if start and end and len(start) == 2 and len(end) == 2:
    #         x1, y1 = map(int, start)
    #         x2, y2 = map(int, end)
    #         adb_shell("input", "swipe", str(x1), str(y1), str(x2), str(y2), "300")
    #     else:
    #         print("drag requires 'start_box' and 'end_box'")
    #         return action_type

    # elif action_type == "press_home":
    #     adb_shell("input", "keyevent", "3")

    # elif action_type == "press_back":
    #     adb_shell("input", "keyevent", "4")

    # elif action_type == "finished":
    #     content = response.get("content", "")
    #     print(f"[FINISHED]: {content}")

    else:
        print(f"Unknown action_type: {action_type}")

    return action_type

def gemma_action(action_type: str, response: dict) -> str:
    
    pass

def run_adb_action(response: dict) -> str:
    """
    서버 응답에 따라 ADB 명령 실행
    """
    print(response)
    
    if "qwen" in response["name"]:
        action_type = qwen_action(response)
        
    elif "uitars" in response["name"]:
        action_type = uitars_action(response)
        
    # elif response["name"] == "gemma":
    #       action_type = gemma_action(response)

    return action_type

def baseline(args):
    
    # previous_action = ""
    for step in range(args.max_steps):
        
        print(f"\nStep {step}: Taking screenshot...")
        image_bytes = take_screenshot(args, step)
        
        print("Sending to server...")
        # print("previous action : ", [previous_action])
        response = send_to_server(args, args.task, image_bytes, step, "baseline", "", args.app_name)
        # response = send_to_server(args, args.task, image_bytes, step, "baseline", str(previous_action), args.app_name)
        
        print("Model Output:", json.dumps(response, indent=2, ensure_ascii=False))
        
        action_type = run_adb_action(response)
        # previous_action = response
        if action_type == "terminate" or action_type == "finished":
            print("Task complete. Exiting.")
            break
        
        time.sleep(3)

    # 마지막 결과 스크린샷
    final_step = step + 1
    take_screenshot(args, final_step)
    print("Final screenshot saved.")
    
def pav(args):
    
    step = 0
    image_bytes = take_screenshot(args, step)
    
    output_history= {}
    
    actor_output = {}
    verifier_output = {}
    
    for step in range(args.max_steps):
        
        print(f"### Step {step+1}")
        
        if step == 0:
            role = "planner"
            response = send_to_server(args, args.task, image_bytes, step, role, "", args.app_name)
            print(">>>Planner Output:", response["macro_action_plan"])
            
            output_history["planner"] = response["macro_action_plan"]

            previous_action = response["arguments"]
            
        else: # step 1, step 2, ...
            
            role = "actor"
            response = send_to_server(args, args.task, image_bytes, step, role, str(previous_action), args.app_name)
            
        print(">>>Actor Output:", json.dumps(response, indent=2, ensure_ascii=False))
        action_type = run_adb_action(response)
        
        actor_output[step] = response["arguments"]
        
        time.sleep(2)
            
        if action_type == "terminate" or action_type == "finished":
            continue
        
        if action_type == "task_completed":
            print("All macro actions are completed.")
            break
        
        image_bytes = take_screenshot(args, step+1)
        
        previous_action = response["arguments"]
        
        next_image_bytes = take_screenshot(args, step+1)
        
        role = "verifier"
        response = send_to_server(args, args.task, next_image_bytes, step, role, "", args.app_name)
        
        verifier_output[step] = response
        
        print(f"\n>>>Verifier Output: {response}")
        
        if response["task_completed"] == -1:
            
            print("All macro actions are completed.")
            break
        
        image_bytes = next_image_bytes
            
        time.sleep(2)
        
    output_history["actor"] = actor_output
    output_history["verifier"] = verifier_output
    
    with open(f"{args.image_path}/output_history.json", "w") as f:
        json.dump(output_history, f, indent=2, ensure_ascii=False)
        
    print(">>>Finished Task.")


def pa(args):
    
    step = 0
    image_bytes = take_screenshot(args, step)
    
    output_history= {}

    actor_output = {}
    
    for step in range(args.max_steps):
        
        print(f"### Step {step+1}")
        
        if step == 0:
            role = "planner"
            response = send_to_server(args, args.task, image_bytes, step, role, "", args.app_name)
            print(">>>Planner Output:", response["macro_action_plan"])

            output_history["planner"] = response["macro_action_plan"]
            macro_action_plan = response["macro_action_plan"]
            previous_action = response["arguments"]     # 1st micro action
        else: # step 1, step 2, ...
            role = "actor"
            response = send_to_server(args, args.task, image_bytes, step, role, str(previous_action), args.app_name)
            
        print(">>>Actor Output:", json.dumps(response, indent=2, ensure_ascii=False))
        action_type = run_adb_action(response)
        
        actor_output[step] = response["arguments"]
        
        time.sleep(2)
        
        current_macro_action = macro_action_plan[0]
        if action_type == "terminate" or action_type == "finished":
            print(f"<{current_macro_action}> completed!")
            
            if len(macro_action_plan) > 1:
                macro_action_plan.pop(0)

            else:
                image_bytes = take_screenshot(args, step+1)
                previous_action = response["arguments"]
                next_image_bytes = take_screenshot(args, step+1)
                print("All macro actions are completed.")
            break
        else:
            print(f"<{current_macro_action}> still in progress!")
        
        if action_type == "task_completed":
            print("All macro actions are completed.")
            break
        
        image_bytes = take_screenshot(args, step+1)
        previous_action = response["arguments"]
        next_image_bytes = take_screenshot(args, step+1)
        image_bytes = next_image_bytes
            
        time.sleep(2)        
            
    output_history["actor"] = actor_output
    
    with open(f"{args.image_path}/output_history.json", "w") as f:
        json.dump(output_history, f, indent=2, ensure_ascii=False)
        
    print(">>>Finished Task.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="VLM Mobile Agent")
    parser.add_argument("--server", type=str, default="http://143.248.158.42:8000/predict", help="Server URL") # loki1: 143.248.158.22 / loki2: 143.248.158.71
    parser.add_argument("--method", type=str, default="baseline", help="Method to use (pav, baseline)")
    parser.add_argument("--task_number", type=str, required=True, help="Task Number")
    parser.add_argument("--task", type=str, required=True, help="Text task to perform")
    parser.add_argument("--image_path", type=str, default="./qwen_3B_baseline_screenshots", help="Path to save screenshots")
    parser.add_argument("--max_steps", type=int, default=10, help="Max number of steps before termination")
    parser.add_argument("--app_name", type=str, default="google_maps", help="App name for planner prompt")

    args = parser.parse_args()
    
    if args.method == "baseline":
        baseline(args)
        
    elif args.method == "pav":
        pav(args)
    
    elif args.method =="pa":
        pa(args)
# ADB path
# export ANDROID_HOME=$HOME/Library/Android/sdk
# export PATH=$PATH:$ANDROID_HOME/platform-tools
# source ~/.zshrc 

# echo 'export ANDROID_HOME=$HOME/Library/Android/sdk' >> ~/.zshrc
# echo 'export PATH=$PATH:$ANDROID_HOME/platform-tools' >> ~/.zshrc
