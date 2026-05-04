#!/bin/bash

set -e

# 1. setup adb tool to use this script -> https://developer.android.com/tools/adb
# 2. add exec permission:
#    chmod +x collect_data.sh
# 3. run it from project root:
#    ./collect_data.sh

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# script is directly in project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

DATA_DIR="$PROJECT_ROOT/data"
ARCHIVE_DIR="$PROJECT_ROOT/dataset_$TIMESTAMP"

SOURCE=/sdcard/Android/data/com.Politechnika.VrEyeGestureAnalysis/files/

echo "Project root: $PROJECT_ROOT"
echo "Target data folder: $DATA_DIR"

# check device connection
DEVICE=$(adb devices | grep -v "List of devices" | grep "device$")

if [ -z "$DEVICE" ]; then
    echo "No headset connected. Please connect your device and try again."
    exit 1
fi

echo "Device found."

# search for .csv files on headset
CSV_FILES=$(adb shell ls "$SOURCE" 2>/dev/null | tr -d '\r' | grep "\.csv$" || true)

if [ -z "$CSV_FILES" ]; then
    echo "No .csv files found in $SOURCE. Use test environment to save data."
    exit 1
fi

echo "Found CSV result files on headset."

# archive existing local data folder if it exists
if [ -d "$DATA_DIR" ]; then
    echo "Existing data folder found."
    echo "Renaming:"
    echo "$DATA_DIR"
    echo "to:"
    echo "$ARCHIVE_DIR"

    mv "$DATA_DIR" "$ARCHIVE_DIR"
fi

# create fresh data folder
mkdir -p "$DATA_DIR"

echo "Starting pull into: $DATA_DIR"

echo "$CSV_FILES" | while read -r file; do
    adb pull "$SOURCE$file" "$DATA_DIR/"
done

COUNT=$(find "$DATA_DIR" -maxdepth 1 -name "*.csv" | wc -l | tr -d ' ')

echo "Pulled $COUNT CSV file(s) to $DATA_DIR"
echo "Done."