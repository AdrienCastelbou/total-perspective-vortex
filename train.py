import mne
import sys
import matplotlib.pyplot as plt
from EEG_Pipeline import EEG_Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import Vectorizer
import numpy as np

def get_epochs():
    if len(sys.argv) != 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 main.py name_of_remote_eeg")
    filename = sys.argv[1]
    raw = mne.read_epochs(filename, preload=True)
    return raw


def main():
    epochs = get_epochs()
    X = epochs.get_data()
    event_mapping = {"Rest": 1, "Match/LF-Fists": 2, "Match/RF-BFeet": 3}
    y = epochs.events[:,-1]
    clf = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        LogisticRegression(solver='lbfgs')  # liblinear is faster than lbfgs
    )
    clf.fit(X, y)
    print(cross_val_score(clf, X, y, cv=10))



if __name__ == "__main__":
    main()