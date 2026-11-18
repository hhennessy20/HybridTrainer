# WESAD (Wearable Stress and Affect Detection) Dataset

## Overview
The WESAD dataset is a publicly available multimodal dataset for wearable stress and affect detection. It records physiological signals from 15 subjects during laboratory stress, amusement, and baseline neutral conditions using both chest-worn (RespiBAN) and wrist-worn (Empatica E4) devices. In HybridTrainer, WESAD is used to ground physiological stress responses (electrodermal activity / EDA elevation and heart rate variability) to calibrate the probability and severity of daily mental stress spikes in the user simulation environment.

## Dataset Details
* **Raw Size:** ~16.37 GB
* **Subjects:** 15 subjects (S2 through S17)
* **Sensors:** RespiBAN chest sensor (ECG, EDA, EMG, respiration, temperature, 3-axis accelerometer at 700 Hz) and Empatica E4 wristband (BVP, EDA, temperature, 3-axis accelerometer).
* **Conditions:** Baseline, Stress (Trier Social Stress Test), Amusement, Meditation.

## Reference & Citation
* **Paper:** Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* In Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18), pp. 400–408.
* **Paper URL:** [https://dl.acm.org/doi/10.1145/3242969.3242985](https://dl.acm.org/doi/10.1145/3242969.3242985)
* **UCI Repository:** [https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
* **Kaggle Dataset:** [https://www.kaggle.com/datasets/uciml/wearable-stress-and-affect-detection](https://www.kaggle.com/datasets/uciml/wearable-stress-and-affect-detection)

## Download Instructions
1. Download `WESAD.zip` from the UCI or Kaggle repository link above.
2. Extract the archive so that subject directories (`S2`, `S3`, etc.) are located at `Data/WESAD/WESAD/`.
3. Run the inspection script:
   ```bash
   python Baseline/data/prepare_data.py --dataset wesad
   ```
