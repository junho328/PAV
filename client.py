import base64, io, subprocess, tempfile, requests, json, time
import argparse
from pathlib import Path
from PIL import Image

def take_screenshot(args, step: int) -> bytes:
    """
    ADB를 사용해 에뮬레이터 화면을 캡처하고 지정된 위치에 저장하며 PNG 바이트를 반환
    """
    output_file = Path(args.image_path) / f"screenshot_{step}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # adb로 임시 스크린샷 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        subprocess.run(
            ["adb", "-s", args.device_id, "exec-out", "screencap", "-p"],
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

def send_to_server(args, task: str, image_bytes: bytes, step: int) -> dict:
    """
    서버로 task + base64 이미지 전송 → 응답 JSON 반환
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"task": task, "image_base64": b64, "step": step}

    r = requests.post(args.server, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def run_adb_action(device_id: str, response: dict) -> str:
    """
    서버 응답에 따라 ADB 명령 실행
    """
    action_type = response.get("action_type", "")
    arguments = response.get("arguments", [])

    def adb_shell(*args):
        subprocess.run(["adb", "-s", device_id, "shell"] + list(args), check=True)

    if action_type == "system_button":
        if not arguments:
            print("⚠️ system_button requires arguments")
            return action_type
        key_map = {
            "HOME": "3",
            "BACK": "4",
            "MENU": "82",
            "ENTER": "66"
        }
        key_code = key_map.get(arguments[0].upper())
        if key_code:
            adb_shell("input", "keyevent", key_code)
        else:
            print(f"Unknown system button: {arguments[0]}")

    elif action_type == "click":
        if len(arguments) < 2:
            print("click requires [x, y]")
            return action_type
        x, y = int(arguments[0]), int(arguments[1])
        adb_shell("input", "tap", str(x), str(y))

    elif action_type == "swipe":
        if len(arguments) < 4:
            print("swipe requires [x1, y1, x2, y2]")
            return action_type
        x1, y1, x2, y2 = map(int, arguments)
        adb_shell("input", "swipe", str(x1), str(y1), str(x2), str(y2))

    elif action_type == "type":
        if not arguments:
            print("type requires [text]")
            return action_type
        text = arguments[0].replace(" ", "%s")
        adb_shell("input", "text", text)

    elif action_type == "long_click":
        if len(arguments) < 2:
            print("long_click requires [x, y]")
            return action_type
        x, y = int(arguments[0]), int(arguments[1])
        adb_shell("input", "swipe", str(x), str(y), str(x), str(y), "1000")

    elif action_type == "key":
        if not arguments:
            print("key requires [key_code]")
            return action_type
        adb_shell("input", "keyevent", str(arguments[0]))

    elif action_type == "open":
        if not arguments:
            print("open requires [package_name]")
            return action_type
        package_name = arguments[0]
        adb_shell("monkey", "-p", package_name, "-c", "android.intent.category.LAUNCHER", "1")

    elif action_type == "wait":
        if not arguments:
            print("wait requires [seconds]")
            return action_type
        seconds = int(arguments[0])
        time.sleep(seconds)

    elif action_type == "terminate":
        status = arguments[0] if arguments else "unknown"
        print(f"Task terminated with status: {status}")

    else:
        print(f"Unknown action_type: {action_type}")

    return action_type

def main(args):
    for step in range(args.max_steps):
        print(f"\nStep {step}: Taking screenshot...")
        image_bytes = take_screenshot(args, step)
        
        print("Sending to server...")
        response = send_to_server(args, args.task, image_bytes, step)
        
        print("Model Output:", json.dumps(response, indent=2, ensure_ascii=False))
        
        action_type = run_adb_action(args.device_id, response)
        
        if action_type == "terminate":
            print("Task complete. Exiting.")
            break
        
        time.sleep(1)

    # 마지막 결과 스크린샷
    final_step = step + 1
    take_screenshot(args, final_step)
    print("Final screenshot saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Mobile Agent")
    parser.add_argument("--server", type=str, default="http://<서버_IP>:8000/predict", help="Server URL")
    parser.add_argument("--device_id", type=str, default="emulator-5554", help="ADB Device ID")
    parser.add_argument("--task", type=str, required=True, help="Text task to perform")
    parser.add_argument("--image_path", type=str, default="images", help="Path to save screenshots")
    parser.add_argument("--max_steps", type=int, default=10, help="Max number of steps before termination")

    args = parser.parse_args()
    main(args)