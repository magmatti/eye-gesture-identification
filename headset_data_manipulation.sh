#!/bin/bash

set -e

# 1. setup adb tool to use this script -> https://developer.android.com/tools/adb
# 2. add exec permission:
#    chmod +x headset_data_manipulation.sh
# 3. run it from project root:
#    ./headset_data_manipulation.sh

SOURCE="/sdcard/Android/data/com.Politechnika.VrEyeGestureAnalysis/files/"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while true; do
    printf "\n1) List CSV files\n2) Collect CSV files\n3) Delete CSV files\n0) Exit\n"

    if ! read -r -p "Choose [0-3]: " CHOICE; then
        printf "\nInput closed. Exiting.\n"
        exit 0
    fi

    case "$CHOICE" in
        0)
            echo "Exiting."
            exit 0
            ;;
        1|2|3)
            ;;
        *)
            echo "Invalid option."
            continue
            ;;
    esac

    if ! command -v adb >/dev/null 2>&1; then
        echo "adb is not installed or not available in PATH."
        continue
    fi

    DEVICE="$(adb devices | awk 'NR > 1 && $2 == "device" { print $1; exit }')"

    if [ -z "$DEVICE" ]; then
        echo "No headset connected. Please connect your device and try again."
        continue
    fi

    if ! REMOTE_FILES="$(adb -s "$DEVICE" shell ls "$SOURCE" 2>/dev/null)"; then
        echo "Could not read $SOURCE on the headset."
        continue
    fi

    CSV_FILES="$(printf "%s\n" "$REMOTE_FILES" | tr -d '\r' | grep -E '\.csv$' || true)"

    if [ -z "$CSV_FILES" ]; then
        echo "No .csv files found in $SOURCE."
        continue
    fi

    case "$CHOICE" in
        1)
            printf "%s\n" "$CSV_FILES"
            ;;
        2)
            DATA_DIR="$SCRIPT_DIR/data/data_$(date +'%Y%m%d_%H%M%S')"

            while [ -e "$DATA_DIR" ]; do
                sleep 1
                DATA_DIR="$SCRIPT_DIR/data/data_$(date +'%Y%m%d_%H%M%S')"
            done

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
done
