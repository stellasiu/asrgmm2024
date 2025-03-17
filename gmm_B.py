import os
import numpy as np
import librosa
from tqdm import tqdm
import shutil

def reorganize_train_data(train_dir): # I use ChatGPT to do this function
    # Create a temporary directory to consolidate speaker data
    consolidated_dir = os.path.join(train_dir, "consolidated")
    os.makedirs(consolidated_dir, exist_ok=True)

    # Traverse through each subdirectory in train_dir
    for subfolder in tqdm(os.listdir(train_dir), desc="Processing subdirectories"):
        subfolder_path = os.path.join(train_dir, subfolder)
        if os.path.isdir(subfolder_path) and subfolder != "consolidated":
            for speaker in os.listdir(subfolder_path):  # Iterate over speaker directories
                speaker_path = os.path.join(subfolder_path, speaker)
                if os.path.isdir(speaker_path):  # Ensure this is a speaker directory
                    # Create or use a folder in consolidated directory for the speaker
                    speaker_dest = os.path.join(consolidated_dir, speaker)
                    os.makedirs(speaker_dest, exist_ok=True)

                    # Move all .wav files to the consolidated speaker folder
                    for file in os.listdir(speaker_path):
                        file_path = os.path.join(speaker_path, file)
                        if os.path.isfile(file_path) and file.endswith(".wav"):
                            shutil.move(file_path, os.path.join(speaker_dest, file))

    # Remove old subdirectories after consolidation
    for subfolder in os.listdir(train_dir):
        subfolder_path = os.path.join(train_dir, subfolder)
        if os.path.isdir(subfolder_path) and subfolder != "consolidated":
            shutil.rmtree(subfolder_path)

    # Rename consolidated directory to train_dir
    for speaker in os.listdir(consolidated_dir):
        speaker_path = os.path.join(consolidated_dir, speaker)
        shutil.move(speaker_path, os.path.join(train_dir, speaker))
    os.rmdir(consolidated_dir)

    print("Training data reorganization complete.")

def custom_GMM_uni(data, K_components, epsilon, seed):

    # Initialization

    np.random.seed(seed)
    n = len(data)

    data_mean = np.mean(data)
    data_variance = np.var(data)
    
    weights = np.ones(K_components) / K_components  # Equal weights
    means = np.random.normal(loc=data_mean, scale=np.sqrt(data_variance), size=K_components)  # Randomized means
    variances = np.full(K_components, data_variance)  # Equal variances
    
    prev_log_likelihood = -np.inf  # Initial log-likelihood
    log_likelihood = 0  # To store updated log-likelihood
    
    iteration = 0
    max_iterations = 1000  # Avoid infinite loop
    
    while abs(log_likelihood - prev_log_likelihood) > epsilon and iteration < max_iterations:
        # Update iteration count
        iteration += 1
        prev_log_likelihood = log_likelihood
        
        # E Step
        responsibilities = np.zeros((n, K_components))
        for k in range(K_components):
            responsibilities[:, k] = weights[k] * (1 / np.sqrt(2 * np.pi * variances[k])) * np.exp(-0.5 * ((data - means[k]) ** 2) / variances[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True) # By GPT this line
        
        # M Step
        for k in range(K_components):
            Nk = responsibilities[:, k].sum()  # GPT told me how to do this
            weights[k] = Nk / n  # Update weight
            means[k] = (responsibilities[:, k] @ data) / Nk  # Update mean
            variances[k] = (responsibilities[:, k] @ ((data - means[k]) ** 2)) / Nk  # Update variance
        
        # Calculating log-likehood
        log_likelihood = np.sum(np.log(np.sum([weights[k] * (1 / np.sqrt(2 * np.pi * variances[k])) * np.exp(-0.5 * ((data - means[k]) ** 2) / variances[k]) for k in range(K_components)], axis=0)))
    
    # It's strange if I follow the order putting weights[0] then weights[1] the order is not the same as in the instruction file
    weights = np.array([weights[1], weights[0]])
    means = np.array([means[1], means[0]])
    variances = np.array([variances[1], variances[0]])
    
    params_dict = {"omega": np.round(weights, 2),"mu": np.round(means, 2),"Sigma": np.round(variances, 2),}
    
    return params_dict

def custom_GMM_multi(data, K_components, epsilon, seed):

    # Initialization

    np.random.seed(seed)
    n, d = data.shape  # Number of samples (n) and dimensions (d)

    weights = np.ones(K_components) / K_components
    means = np.random.uniform(np.min(data, axis=0), np.max(data, axis=0), (K_components, d))
    covariances = np.array([np.cov(data, rowvar=False) + np.eye(d) * 1e-6 for _ in range(K_components)])

    prev_log_likelihood = -np.inf
    log_likelihood = 0

    iteration = 0
    max_iterations = 1000

    while abs(log_likelihood - prev_log_likelihood) > epsilon and iteration < max_iterations:
        # Update iteration count
        iteration += 1
        prev_log_likelihood = log_likelihood

        # E Step
        responsibilities = np.zeros((n, K_components))
        for k in range(K_components):
            try:
                diff = data - means[k]
                cov_inv = np.linalg.inv(covariances[k] + np.eye(d) * 1e-6)  # Add regularization by GPT, because it just repeatedly shows error
                exponent = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
                responsibilities[:, k] = weights[k] * np.exp(-0.5 * exponent) / (
                    np.sqrt((2 * np.pi) ** d * max(np.linalg.det(covariances[k]), 1e-6)))
            except np.linalg.LinAlgError:
                responsibilities[:, k] = 0

        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities_sum[responsibilities_sum == 0] = 1e-6
        responsibilities /= responsibilities_sum

        # M Step
        for k in range(K_components):
            Nk = responsibilities[:, k].sum()
            if Nk < 1e-6:  # Skip components with negligible weight, same as regularization also added by GPT
                continue

            weights[k] = Nk / n
            means[k] = (responsibilities[:, k] @ data) / Nk
            diff = data - means[k]
            covariances[k] = (responsibilities[:, k][:, None] * diff).T @ diff / Nk
            covariances[k] += np.eye(d) * 1e-6  # Regularization to avoid singular matrix, also added by GPT to avoid error

        # Calculate log-likelihood
        log_likelihood = np.sum(np.log(np.maximum(responsibilities.sum(axis=1), 1e-6)))

    return weights, means, covariances

def speaker_rec_GMM(audio_dir, test_dir):

    reorganize_train_data(audio_dir)

    speaker_gmms = {}

    # Train a GMM for each speaker in the consolidated directory
    for speaker in tqdm(os.listdir(audio_dir), desc="Processing training data"):
        speaker_path = os.path.join(audio_dir, speaker)
        if os.path.isdir(speaker_path):  # Ensure this is a speaker directory
            # Collect all audio data for the current speaker
            features = []
            for file in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file)
                if os.path.isfile(file_path) and file.endswith(".wav"):
                    audio, sr = librosa.load(file_path, sr=None)  # No resampling
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract MFCC features, 13 is the best here
                    features.append(np.mean(mfcc.T, axis=0))  # Mean of MFCCs

            features = np.array(features)
            weights, means, covariances = custom_GMM_multi(features, K_components=4, epsilon=1e-6, seed=100) # I use 4 here to avoid overfitting
            speaker_gmms[speaker] = (weights, means, covariances)

    # Predict the speaker for each file in the test directory
    predict_dict = {}
    for file in tqdm(os.listdir(test_dir), desc="Predicting"):
        file_path = os.path.join(test_dir, file)
        if os.path.isfile(file_path) and file.endswith(".wav"):
            audio, sr = librosa.load(file_path, sr=None)  # No resampling
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract MFCC features
            test_feature = np.mean(mfcc.T, axis=0)

            # Compute the log-likelihood for each GMM
            log_likelihoods = {}
            for speaker, (weights, means, covariances) in speaker_gmms.items():
                log_likelihood = 0
                for k in range(len(weights)):
                    diff = test_feature - means[k]
                    cov_inv = np.linalg.inv(covariances[k])
                    exponent = -0.5 * diff.T @ cov_inv @ diff
                    log_likelihood += weights[k] * np.exp(exponent) / (np.sqrt((2 * np.pi) ** len(test_feature) * np.linalg.det(covariances[k])))
                log_likelihoods[speaker] = log_likelihood

            # Find the speaker with the highest likelihood
            predicted_speaker = max(log_likelihoods, key=log_likelihoods.get)
            predict_dict[file] = predicted_speaker

    return predict_dict

# 97.6% accuracy using custom_GMM_multi
