# Eye Gesture Analysis

Project for eye gestures identification based on data from a Meta Quest Pro headset.

The project detects eye gesture events inside recordings with a threshold-based model.

## Functionality overview

1. Loads Unity CSV files.
2. Converts Unity eye quaternions into gaze direction vectors.
3. Computes binocular gaze speed in degrees per second.
4. Computes Meta blink average from left/right blink weights.
5. Applies fixation, saccade, and blink thresholds.
6. Extracts and counts detected gesture events.
7. Saves processed CSVs, event tables, summaries, and plots.

## Create environment

```bash
conda create --name eye-gesture-identification python=3.11
conda activate eye-gesture-identification
pip install -r requirements.txt
```

## Collect data

The `collect_data.sh` script pulls saved `.csv` recordings from a connected Meta Quest headset into the local `data/` folder.

Before using it:

1. Install and configure `adb`: https://developer.android.com/tools/adb
2. Connect the headset and allow USB debugging.
3. Make the script executable:

```bash
chmod +x collect_data.sh
```

Run it from the project root:

```bash
./collect_data.sh
```

If a `data/` folder already exists, the script renames it to `dataset_YYYYMMDD_HHMMSS` before creating a fresh `data/` folder.

## Run project

```bash
python -m src.analyze --data data --output reports
```
