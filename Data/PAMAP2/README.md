# PAMAP2 Physical Activity Monitoring Dataset

## Overview
The PAMAP2 Physical Activity Monitoring dataset contains continuous biometric and motion sensor recordings from 9 subjects performing 18 physical activities (ranging from sedentary states such as lying and sitting, to vigorous activities such as running, cycling, and rope jumping). In HybridTrainer, PAMAP2 is used to empirically map physical exertion levels to heart rate distributions and ground the Borg CR10 perceived exertion scale ($RPE \in [1, 10]$).

## Dataset Details
* **Raw Size:** ~1.61 GB
* **Subjects:** 9 healthy subjects (1 female, 8 male)
* **Sensors:** 3 Inertial Measurement Units (IMUs) over chest, wrist, and dominant ankle (sampling frequency 100 Hz), plus HR monitor (sampling frequency ~9 Hz).
* **Activities:** 18 activities (e.g., lying, sitting, standing, walking, running, cycling, Nordic walking, ascending/descending stairs, vacuum cleaning, rope jumping).

## Reference & Citation
* **Paper:** Reiss, A., & Stricker, D. (2012). *Creating and Benchmarking a New Dataset for Physical Activity Monitoring.* In Proceedings of the 5th International Conference on Pervasive Technologies Related to Assistive Environments (PETRA '12), pp. 1–8.
* **Paper URL:** [https://dl.acm.org/doi/10.1145/2413097.2413148](https://dl.acm.org/doi/10.1145/2413097.2413148)
* **UCI Repository:** [https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)

## Download Instructions
1. Download `PAMAP2_Dataset.zip` from the UCI repository link above.
2. Extract the archive into this directory so that the protocol directory is located at `Data/PAMAP2/PAMAP2_Dataset/Protocol/`.
3. Run the inspection script:
   ```bash
   python Baseline/data/prepare_data.py --dataset pamap2
   ```
