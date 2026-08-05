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
8. Compares detected saccades against the target movements recorded by the Unity app (`TargetRotX`/`TargetRotY`).
9. Compares detected blink counts against the metronome beeps of the Unity blink scenario.
10. Prints a compact report aggregated per participant and scenario in the console.
11. The Jupyter notebooks additionally offer detailed per-file tables, the full event list and plots.

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

0. Exit script.
1. List `.csv` files stored on the headset.
2. Collect and save `.csv` files into `/data` folder as `data_YYYYMMDD_HHMMSS`.
3. Delete all captured .csv files from headset.

Example workflow should look like this:

1. Running the script.
2. Listing all files stored on Meta Quest Pro.
3. Collecting .csv into `/data/data_YYYYMMDD_HHMMSS` directory.
4. Renaming directory as `/data/participant_name` or any other suitable name.
5. Deleting all files stored on headset in order to declutter the storage.
6. Exiting the script.

## Run project

For identification overview run project from the console.

```bash
uv run -m src.analyze
```

For more information and "in depth analysis output" use `identification_output.ipynb` and `dataset_plots.ipynb`.

Set `VERBOSE = True` in `identification_output.ipynb` in order to see full per-file tables and event list.
