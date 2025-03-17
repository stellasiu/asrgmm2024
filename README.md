# Speaker Recognition using GMMs

This project implements a **Gaussian Mixture Model (GMM) based Speaker Recognition System**.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Results](#results)

## Overview
This repository contains implementations for training and testing **GMM-based speaker recognition** models. It supports both **univariate** and **multivariate** GMMs and provides a method to reorganize training data for better performance.

The implementation follows the requirements from the assignment description and achieves high accuracy (>95%) in speaker prediction.

## Features
- **Speaker Identification using GMMs**: Identifies speakers from test audio files based on trained GMMs.
- **Data Preprocessing**: Automatically restructures training data for improved performance.
- **MFCC Feature Extraction**: Uses **Mel-Frequency Cepstral Coefficients (MFCCs)** to extract features from speech.
- **Univariate and Multivariate GMMs**: Implements both custom univariate and multivariate GMMs.
- **Progress Tracking**: Uses `tqdm` for progress tracking during training and prediction.

## File Structure
```
|-- gmm_A.py      # Implements GMM-based speaker recognition
|-- gmm_B.py      # Implements custom univariate and multivariate GMMs
|-- README.md          # Project documentation
```

## Installation
To run this project, install the required dependencies:

```bash
pip install numpy scipy librosa tqdm scikit-learn
```

Ensure you have **Python 3.7+** installed.

## Usage
### Running Speaker Recognition (Part A)
```python
from gmm_A import speaker_rec_GMM

train_data_dir = "./train_data"
test_data_dir = "./test_data"
predictions = speaker_rec_GMM(train_data_dir, test_data_dir)
print(predictions)
```

### Running Custom GMMs (Part B)
#### Univariate GMM
```python
from gmm_B import custom_GMM_uni
import numpy as np

data = np.array([4.6, 12.4, 10.2, 12.8, 12.3])
params = custom_GMM_uni(data, K_components=2, epsilon=1e-6, seed=1234)
print(params)
```

#### Multivariate GMM
```python
from gmm_B import custom_GMM_multi

data = np.random.rand(100, 3)  # Example data with 3 variables
params = custom_GMM_multi(data, K_components=3, epsilon=1e-6, seed=1234)
print(params)
```

## Implementation Details
- **Speaker Recognition (Part A)**:
  - Restructures training data into speaker-specific folders.
  - Extracts **MFCC features** from audio.
  - Trains a **Gaussian Mixture Model (GMM)** for each speaker.
  - Predicts speaker labels for test audio.

- **Custom GMM Implementations (Part B)**:
  - `custom_GMM_uni()`: Implements a **univariate GMM**.
  - `custom_GMM_multi()`: Implements a **multivariate GMM**.
  - Avoids **numerical instability** by using **regularization**.
  - Uses **Expectation-Maximization (EM) algorithm** for optimization.

## Results
- Achieved **98.8% accuracy** in speaker recognition using `sklearn`'s GMM.
- Achieved **97.6% accuracy** using a **custom multivariate GMM**.
- **Data restructuring improved accuracy from ~82% to ~98%**.
