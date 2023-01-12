import mne
import sys
import matplotlib.pyplot as plt
from EEG_Pipeline import EEG_Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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


def evaluate_model(name, pipeline, X, y):
    scores = cross_val_score(pipeline, X, y, cv=10)
    print(f"{name} score : {scores.mean()} accuracy with a standard deviation of {scores.std()}")


def main():
    datas_epochs = load_datas()
    X = np.concatenate([epochs.get_data() for epochs in datas_epochs])
    y = np.concatenate([epochs.events[:,-1] for epochs in datas_epochs])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    kn = make_pipeline(Vectorizer(), StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    lr = make_pipeline( Vectorizer(), StandardScaler(), LogisticRegression(solver="lbfgs"))
    svc = make_pipeline( Vectorizer(), StandardScaler(), SVC())
    mlp = make_pipeline( Vectorizer(), StandardScaler(), MLPClassifier(solver="lbfgs"))
    #kn.fit(X_train, y_train)
    #lr.fit(X_train, y_train)
    #svc.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    evaluate_model("mlp", mlp, X, y)
    return
    evaluate_model("Kn", kn, X, y)
    evaluate_model("Lr", lr, X, y)
    evaluate_model("SVC", svc, X, y)




if __name__ == "__main__":
    main()