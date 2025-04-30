import subprocess

def take_screenshot(output_path='screenshot.png'):
    """ 현재 연결된 디바이스에서 스크린샷을 찍어 저장하는 함수 """
    try:
        # adb devices 명령으로 연결된 디바이스 확인
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        devices = [line.split()[0] for line in result.stdout.splitlines() if '\tdevice' in line]

        if not devices:
            print("❌ 연결된 디바이스가 없습니다. adb devices를 확인하세요.")
            return

        # 첫 번째 연결된 디바이스에 대해 스크린샷 실행
        device = devices[0]
        print(f"✅ 디바이스 {device} 에 스크린샷 요청합니다.")

        # 스크린샷 명령 실행
        with open(output_path, 'wb') as f:
            subprocess.run(['adb', '-s', device, 'exec-out', 'screencap', '-p'], stdout=f)

        print(f"📸 스크린샷 저장 완료: {output_path}")

    except Exception as e:
        print(f"에러 발생: {e}")

# 실행 예시
take_screenshot('current_device_screenshot.png')