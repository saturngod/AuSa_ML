import os
import pandas as pd
import numpy as np
import tqdm
import librosa

from sklearn.model_selection import train_test_split

label2int = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

num_samples = None
num_features = None
time_steps = 1

def load_data(vector_length=128):
    """
        Extract feature vectors from audio and save it in .npy format.
        The main idea is if there is no update in data, there is not need to
            extract the feature from audio again and processing will be much faster!
    """
    if not os.path.isdir("results"):
        os.mkdir("results")
        
    if os.path.isfile("results/features.npy") and os.path.isfile("results/labels.npy"):
        X = np.load("results/features.npy")
        y = np.load("results/labels.npy")
        return X, y
    
    df = pd.read_csv('datasets/audio-sentiment.csv')
    df = df.sample(frac=1)
    n_samples = len(df) # total samples
    
    positive_sample = len(df[df['class'] == 'positive'])
    negative_sample = len(df[df['class'] == 'negative'])
    neutral_sample = len(df[df['class'] == 'neutral'])
    print("Total neutral samples:", neutral_sample)
    print("Total positive samples:", positive_sample)
    print("Total negative samples:", negative_sample)
    
    X = np.zeros((n_samples, vector_length))
    y = np.zeros((n_samples, 1))
    
    for i, (filename, classes) in tqdm.tqdm(enumerate(zip(df['filename'], df['class'])), "Loading data", total=n_samples):
        features = extract_feature('datasets/all/'+filename, mfcc=True).reshape(1, -1) 
        X[i] = features
        y[i] = label2int[classes]
        
    np.save("results/features", X)
    np.save("results/labels", y)
    return X, y

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result
    
def split_data(X, y, test_size=0.2, valid_size=0.2):
    
    global num_samples, num_features, time_steps

    num_samples = X.shape[0]
    num_features = X.shape[1]

    X = np.reshape(X, (num_samples, time_steps, num_features))
    

    # split training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=10)
    
    # split training set and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=10)
    # return a dictionary of values
    
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "time_steps": time_steps,
        "n_samples": num_samples,
        "n_features": num_features
    }

def get_result(pred):
    return [k for k, v in label2int.items() if v == pred]