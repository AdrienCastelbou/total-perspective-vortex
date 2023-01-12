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
import os

def load_datas():
    datas_epochs = []
    if len(sys.argv) != 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 main.py path/to/file.edf (can be a directory)")
    path = sys.argv[1]
    if os.path.isfile(path):
        datas_epochs.append(mne.read_epochs(path, preload=True))
    else:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for f in files:
            datas_epochs.append(mne.read_epochs(path + "/" + f, preload=True))
    return datas_epochs

def get_epochs():
    if len(sys.argv) != 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 main.py name_of_remote_eeg")
    filename = sys.argv[1]
    raw = mne.read_epochs(filename, preload=True)
    return raw


def main():
    datas_epochs = load_datas()
    X = np.concatenate([epochs.get_data() for epochs in datas_epochs])
    y = np.concatenate([epochs.events[:,-1] for epochs in datas_epochs])
    clf = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        LogisticRegression(solver='lbfgs')  # liblinear is faster than lbfgs
    )
    clf.fit(X, y)
    scores = cross_val_score(clf, X, y, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))



if __name__ == "__main__":
    main()