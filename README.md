# Eye Gesture Analysis

Project for eye gestures indentification based of data got from Meta Quest Pro headset.

The baseline is inspired by classic eye-tracking event detection ideas:
- **fixation / saccade**: velocity + duration + dispersion logic (I-VT / I-DT style)
- **blink**: binocular hysteresis thresholding on blink weights

## Functionality overview

1. Loads your Unity CSV files.
2. Converts Unity eye quaternions into gaze direction vectors.
3. Computes yaw, pitch, angular speed, dispersion, and target-following metrics.
4. Runs gesture-specific detectors.
5. Produces a simple **trial-level classification** for the controlled experiment.
6. Saves a report to `reports/rule_based_results.csv`.

## Recommended environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
 python evaluate.py
```

## Notes

- The detector is designed for the **controlled experiment** where each CSV is a single trial.
