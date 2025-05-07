#!/usr/bin/env bash

# ----------------------
# Configuration
# ----------------------
# 경로와 파일 이름을 필요에 맞게 수정하세요.

TASK_APP="google_maps"  
METHOD="pav" # "baseline"
IMAGE_BASE_PATH="/Users/junhona/mobile_agent/screeenshots/$METHOD/$TASK_APP"

TASK_FILE="./task/google_maps_tasks.txt" 
EMULATOR_CMD="$HOME/Library/Android/sdk/emulator/emulator"
ADB_CMD="$HOME/Library/Android/sdk/platform-tools/adb"
AVD_NAME="Pixel_9"
SNAPSHOT_NAME="google_maps"
PAV_CLIENT_SCRIPT="./client/pav_client.py"

# 실제 실행할 앱의 패키지 이름 (TASK_FILE에 나오는 Task 명령어와 별개)
APP_NAME="com.google.android.apps.maps"

# ----------------------
# Functions
# ----------------------
start_emulator() {
  echo "Starting emulator $AVD_NAME with snapshot $SNAPSHOT_NAME..."
  "$EMULATOR_CMD" -avd "$AVD_NAME" -snapshot "$SNAPSHOT_NAME" -no-boot-anim -no-snapshot-save > /dev/null 2>&1 &
  EMU_PID=$!
  echo "Emulator PID: $EMU_PID"
  # Wait for device to be ready
  "$ADB_CMD" wait-for-device
  echo "Waiting for boot to complete..."
  until [[ $("$ADB_CMD" shell getprop sys.boot_completed | tr -d '\r') == "1" ]]; do
    sleep 1
  done
  echo "Emulator booted."
}

stop_emulator() {
  echo "Stopping emulator..."
  "$ADB_CMD" emu kill
  wait $EMU_PID 2>/dev/null || true
  echo "Emulator stopped."
}

# ----------------------
# Main Loop
# ----------------------
INDEX=0
# FD 3을 TASK_FILE로 연결
exec 3< "$TASK_FILE"

while IFS= read -r TASK <&3; do
  [[ -z "$TASK" ]] && continue
  echo "---- task #$INDEX: $TASK"

  start_emulator
  "$ADB_CMD" shell monkey -p "$APP_NAME" -c android.intent.category.LAUNCHER 1

  sleep 10

  CURRENT_IMAGE_PATH="$IMAGE_BASE_PATH/$INDEX"

  python "$PAV_CLIENT_SCRIPT" \
    --method "$METHOD" \
    --task "$TASK" \
    --image_path "$CURRENT_IMAGE_PATH" \
    --app_name "$TASK_APP" \
    || echo "⚠️ PAV client failed on task #$INDEX"

  stop_emulator

  INDEX=$((INDEX+1))
done

# FD 3 닫기
exec 3<&-
echo "All test tasks completed."