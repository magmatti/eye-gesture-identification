# Eye Gesture Analysis

Project for eye tracking events identification based on data from a Meta Quest Pro headset.

It detects and classifies three eye tracking events: fixations, saccades and blinks. It is based on the velocity-threshold method (I-VT) described by [Salvucci & Goldberg (2000)](https://www.cs.drexel.edu/~dds26/publications/Salvucci-ETRA00.pdf), extended with saccade direction classification. Detected events are compared against the target movements and blink cues recorded by the companion [Unity application](https://github.com/magmatti/vr-eye-gesture-analysis), which makes it possible to measure how accurately events are recognised.

## Functionality overview

1. Reads the `.csv` recordings produced by the Unity application.
2. Converts Unity eye rotation quaternions into a binocular gaze direction and its angular speed.
3. Labels samples as a fixation, a saccade or a blink using velocity and blink weight threshold.
4. Groups labelled samples into events and classifies the direction of each saccade as `left`, `right`, `up` or `down`.
5. Compares detected events against the target jumps for saccades and beep sound for blinks.
6. Reports detection quality per participant and scenario: precision, recall, F1, reaction latency and amplitude error.

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
2. Collect and save `.csv` files into `data/` folder as `data_YYYYMMDD_HHMMSS`.
3. Delete all captured `.csv` files from headset.

Example workflow should look like this:

1. Running the script.
2. Listing all files stored on Meta Quest Pro.
3. Collecting `.csv` into `data/data_YYYYMMDD_HHMMSS` directory.
4. Renaming directory as `data/participant_name` or any other suitable name.
5. Deleting all files stored on headset in order to declutter the storage.
6. Exiting the script.

## Run project

For a quick summary of the results in the console, run from the project root:

```bash
uv run -m src.analyze
```

The complete set of result tables is available in `identification_output.ipynb`, and all plots in `dataset_plots.ipynb`.
