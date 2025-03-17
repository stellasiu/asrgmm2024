import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
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

def speaker_rec_GMM(audio_dir, test_dir):
    # Reorganize training data before processing otherwise the performance was very bad
    reorganize_train_data(audio_dir)

    speaker_gmms = {}

    # Train a GMM for each speaker in the consolidated directory
    for speaker in tqdm(os.listdir(audio_dir), desc="Processing training data"):
        speaker_path = os.path.join(audio_dir, speaker)
        if os.path.isdir(speaker_path):
            features = []
            for file in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file)
                if os.path.isfile(file_path) and file.endswith(".wav"):
                    audio, sr = librosa.load(file_path, sr=None)  # No resampling
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=39)  # Extract MFCC features
                    features.append(np.mean(mfcc.T, axis=0))  # Mean of MFCCs

            gmm = GaussianMixture(n_components=16, covariance_type="diag", random_state=100)
            gmm.fit(features)
            speaker_gmms[speaker] = gmm

    # Predict the speaker for each file in the test directory
    predict_dict = {}
    for file in tqdm(os.listdir(test_dir), desc="Predicting"):
        file_path = os.path.join(test_dir, file)
        if os.path.isfile(file_path) and file.endswith(".wav"):
            audio, sr = librosa.load(file_path, sr=None)  # No resampling
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=39)  # Extract MFCC features
            test_feature = np.mean(mfcc.T, axis=0).reshape(1, -1)  # Mean

            # Compute the log-likelihood for each GMM
            log_likelihoods = {speaker: gmm.score(test_feature) for speaker, gmm in speaker_gmms.items()}

            # Find the speaker with the highest likelihood
            predicted_speaker = max(log_likelihoods, key=log_likelihoods.get)
            predict_dict[file] = predicted_speaker

    return predict_dict

# I have 98.8% for now, without consolidating the data I got maximum 82%
# Credit to Jiashu who taught me about number of GMM I needed to process
