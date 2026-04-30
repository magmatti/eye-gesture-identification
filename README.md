# Eye Gesture Analysis

Project for eye gestures identification based on data from a Meta Quest Pro headset.

The baseline is inspired by classic eye-tracking event detection ideas:
- **fixation / saccade**: velocity + duration + dispersion logic (I-VT / I-DT)
- **blink**: binocular hysteresis thresholding on blink weights

## Functionality overview

1. Loads Unity CSV files.
2. Converts Unity eye quaternions into gaze direction vectors.
3. Computes yaw, pitch, angular speed, and dispersion.
4. Runs blink, fixation, and saccade detectors.
5. Produces a **trial-level classification** for the three supported gestures.

## Recommended environment

```bash
conda create --name eye-gesture-identification python=3.11
conda activate eye-gesture-identification
pip install -r requirements.txt
```
## Run

```bash
PYTHONPATH=src python evaluate.py
```

## Notes

- The detector is designed for the **controlled experiment** where each CSV is a single trial.
- The main evaluation focuses on blink, fixation, and saccade trials. Other CSV files can stay in `data/`, but they are excluded from the generated report.
