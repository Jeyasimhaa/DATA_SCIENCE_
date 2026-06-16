import librosa
import numpy as np

def extract_audio_features(audio_path):

    y, sr = librosa.load(audio_path)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc.reshape(1,40)