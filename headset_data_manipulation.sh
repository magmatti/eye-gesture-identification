#!/bin/bash

set -e

# 1. setup adb tool to use this script -> https://developer.android.com/tools/adb
# 2. add exec permission:
#    chmod +x headset_data_manipulation.sh
# 3. run it from project root:
#    ./headset_data_manipulation.sh

SOURCE="/sdcard/Android/data/com.Politechnika.VrEyeGestureAnalysis/files/"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
ARCHIVE_DIR="$SCRIPT_DIR/dataset_$(date +'%Y%m%d_%H%M%S')"

printf "1) List CSV files\n2) Collect CSV files\n3) Delete CSV files\n"
read -r -p "Choose [1-3]: " CHOICE

case "$CHOICE" in
    1|2|3) ;;
    *) echo "Invalid option."; exit 1 ;;
esac

command -v adb >/dev/null 2>&1 || {
    echo "adb is not installed or not available in PATH."
    exit 1
}

DEVICE="$(adb devices | awk 'NR > 1 && $2 == "device" { print $1; exit }')"

if [ -z "$DEVICE" ]; then
    echo "No headset connected. Please connect your device and try again."
    exit 1
fi

REMOTE_FILES="$(adb -s "$DEVICE" shell ls "$SOURCE" 2>/dev/null)" || {
    echo "Could not read $SOURCE on the headset."
    exit 1
}

CSV_FILES="$(printf "%s\n" "$REMOTE_FILES" | tr -d '\r' | grep -E '\.csv$' || true)"

if [ -z "$CSV_FILES" ]; then
    echo "No .csv files found in $SOURCE."
    [ "$CHOICE" = 2 ] && exit 1 || exit 0
fi

case "$CHOICE" in
    1)
        printf "%s\n" "$CSV_FILES"
        ;;
    2)
        if [ -d "$DATA_DIR" ]; then
            echo "Archiving existing data to $ARCHIVE_DIR"
            mv "$DATA_DIR" "$ARCHIVE_DIR"
        fi

        mkdir -p "$DATA_DIR"

        while IFS= read -r FILE; do
            adb -s "$DEVICE" pull "$SOURCE$FILE" "$DATA_DIR/"
        done <<< "$CSV_FILES"

        COUNT="$(find "$DATA_DIR" -maxdepth 1 -name "*.csv" | wc -l | tr -d " ")"
        echo "Pulled $COUNT CSV file(s) to $DATA_DIR"
        ;;
    3)
        read -r -p "Delete all .csv files from the headset? [y/n] " RESPONSE

        case "$RESPONSE" in
            [yY])
                adb -s "$DEVICE" shell "rm ${SOURCE}*.csv"
                echo "Deleted all .csv files from $SOURCE."
                ;;
            *)
                echo "Cancelled."
                ;;
        esac
        ;;
esac
