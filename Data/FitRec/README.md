# FitRec (Endomondo) Dataset

## Overview
The FitRec dataset contains workout records extracted from the Endomondo fitness tracking platform, providing sequential GPS, timestamp, heart rate, speed, and distance measurements across diverse sports (running, cycling, walking). In HybridTrainer, this dataset is used to calibrate empirical workout duration distributions, aerobic heart rate priors, and sport modality profiles.

## Dataset Details
* **Raw Size:** ~12.05 GB (167,373 workouts across 956 users)
* **Modalities:** GPS coordinates, heart rate (bpm), derived speed, distance, elevation
* **Preprocessed Subset:** Evaluated in `Baseline/data/fitrec.py` and `Baseline/data/prepare_data.py`

## Reference & Citation
* **Paper:** Ni, J., Muhlstein, L., & McAuley, J. (2019). *Modeling Heart Rate and Activity Data for Personalized Fitness Recommendation.* In Proceedings of the World Wide Web Conference (WWW '19), pp. 1343–1353.
* **Paper URL:** [https://dl.acm.org/doi/10.1145/3308558.3313643](https://dl.acm.org/doi/10.1145/3308558.3313643)
* **Dataset Repository:** [https://sites.google.com/eng.ucsd.edu/fitrec-project/home](https://sites.google.com/eng.ucsd.edu/fitrec-project/home)
* **Kaggle Dataset:** [https://www.kaggle.com/datasets/tientd95/fitrec-dataset](https://www.kaggle.com/datasets/tientd95/fitrec-dataset)

## Download Instructions
1. Download `endomondoHR_proper.json` from the Dataset Repository or Kaggle Dataset links above. NOTE: Official Dataset Repository link is not working as of late September 2026. Kaggle dataset only contains unprocessed data, which may encounter issues during evaluation. Reach out to repo owner if you need the processed version.
2. Place `endomondoHR_proper.json` inside this directory (`Data/FitRec/`).
3. Run the inspection script:
   ```bash
   python Baseline/data/prepare_data.py --dataset fitrec
   ```
