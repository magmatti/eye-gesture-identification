#!/bin/bash

# 1. setup adb tool to use this script -> https://developer.android.com/tools/adb
# 2. add exec permission:
#    chmod +x delete_data.sh
# 3. run it from project root:
#    ./delete_data.sh

set -e

SOURCE=/sdcard/Android/data/com.Politechnika.VrEyeGestureAnalysis/files/

DEVICE=$(adb devices | grep -v "List of devices" | grep "device$" || true)

if [ -z "$DEVICE" ]; then
    echo "No headset connected. Please connect your device and try again."
    exit 1
fi

CSV_FILES=$(adb shell ls "$SOURCE" 2>/dev/null | tr -d '\r' | grep "\.csv$" || true)

if [ -z "$CSV_FILES" ]; then
    echo "No .csv files found in $SOURCE."
    exit 0
fi

read -r -p "Delete all .csv files from the headset? [y/n] " RESPONSE

case "$RESPONSE" in
    [yY]) adb shell "rm ${SOURCE}*.csv" ;;
    *) echo "Cancelled."; exit 0 ;;
esac

echo "Deleted all .csv files from $SOURCE."
