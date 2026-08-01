# Eye Gesture Analysis

Project for eye gestures identification based on data from a Meta Quest Pro headset.

The project detects eye gesture events inside recordings with a threshold-based model.

## Functionality overview

1. Loads Unity CSV files.
2. Converts Unity eye quaternions into gaze direction vectors.
3. Computes binocular gaze speed in degrees per second.
4. Computes Meta blink average from left/right blink weights.
5. Applies fixation, saccade, and blink thresholds.
6. Extracts fixation, saccade, and blink events.
7. Classifies saccade's move direction as `left`, `right`, `up`, `down`, or `unknown`.
8. Prints event tables and summaries in the console.
9. The same tables plus plots are viewed interactively in the Jupyter notebooks.

## Install uv

Use official documentation: [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

Or if on macOS with homebrew installed:

```bash
brew install uv
```

## Set up the project

From the project root, synchronize the environment:

```bash
uv sync
```

## Manage headset data

The `headset_data_manipulation.sh` script manages `.csv` recordings stored on a connected Meta Quest Pro headset.

Before using it, install and configure [Android Debug Bridge (adb)](https://developer.android.com/tools/adb), connect the headset, allow USB debugging, and make the script executable:

```bash
chmod +x headset_data_manipulation.sh
```

Run it from the project root:

```bash
./headset_data_manipulation.sh
```

Choose an operation from the menu:

1. List `.csv` files stored on the headset.
2. Collect them into `data/`; if `data/` folder already exists it is archived as `dataset_YYYYMMDD_HHMMSS`.
3. Delete all captured .csv files from headset.

## Run project

```bash
uv run -m src.analyze
```
