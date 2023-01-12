import mne
import sys
import matplotlib.pyplot as plt
from EEG_Pipeline import EEG_Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mne.decoding import Vectorizer
import numpy as np
import os
from sklearn.metrics import accuracy_score


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        LogisticRegression(solver='lbfgs',max_iter=1000)  # liblinear is faster than lbfgs
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(accuracy_score(y_test,preds))
    scores = cross_val_score(clf, X, y, cv=10)
    print(scores)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))



if __name__ == "__main__":
    main()