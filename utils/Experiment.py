from process.Preprocess import Preprocess
from utils.experiments import experiments
import os.path
import mne
from mne.decoding import CSP
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from utils.CustomCSP import CustomCSP
from utils.SingleClassTransf import SingleClassTransf
from utils.annotations import annotations
import time
from sklearn.model_selection import train_test_split

class Experiment:
    def __init__(self, subject, run, getAllRuns = False) -> None:
        self.task        = None
        self.filenames   = []
        self.run         = None 
        self.model       = None
        self.subject     = subject
        self.run         = run
        self.get_paths(getAllRuns)


    def get_model(self):
        modelFile = f"models/S{self.subject}{self.task}"
        if os.path.isfile(modelFile):
            with open(modelFile, 'rb') as f:
                self.model = pickle.load(f)
            return True
        return False
        

    def get_paths(self, getAllRuns):
        runs = []
        for task in experiments:
            if self.run in experiments[task]:
                if getAllRuns:
                    runs = experiments[task]
                else:
                    runs.append(self.run)
                self.task = task
                break
        if runs == []:
            raise Exception(f"Run {self.run} of subject {self.subject} is not valable, please use a number between 3 and 14")
        subjectId = self.subject.zfill(3)
        for r in runs:
            runId = r.zfill(2)
            filename = f"./physionet.org/files/eegmmidb/1.0.0/S{subjectId}/S{subjectId}R{runId}.edf"
            if os.path.isfile(filename):
                self.filenames.append(filename)
            elif r == self.run:
                raise Exception(f"Run {self.run} of subject {self.subject} does not exist, make sure the corresponding file exists in ./physionet.org/files/eegmmidb/1.0.0/SXXX/SXXXRXX.edf")


    def preprocess_data(self, vizualize=False):
        preprocessor = Preprocess(vizualize, annotations[self.task])
        raw = preprocessor.run(self.filenames)
        return raw

    def get_data(self, raw):
        X = None
        y = None
        events, event_id = mne.events_from_annotations(raw)

        if len(event_id) < 2:
            events = mne.make_fixed_length_events(raw, next(iter(event_id.values())), duration=4) 
        tmin, tmax = -1.0, 4.0
        picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, exclude="bads")
        epochs = mne.Epochs(raw, events,tmin=tmin, tmax=tmax, picks=picks, proj=True, baseline=None, preload=True)
        epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
        X = epochs_train.get_data()
        y = epochs.events[:, -1]
        return X, y

    def train(self):
        raw = self.preprocess_data()
        X, y = self.get_data(raw)
        X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=7)
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        scores = []

        lda = LinearDiscriminantAnalysis()
        if len(np.unique(y)) == 1:
            transformer = SingleClassTransf()
            clf = Pipeline([("SingleClassTransf", transformer), ("LDA", lda)])
        else:
            customCsp = CustomCSP()
            clf = Pipeline([("CSP", customCsp), ("LDA", lda)])

        clf.fit(X_train, y_train)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, n_jobs=None)
        self.model = clf
        accuracy = accuracy_score(self.model.predict(X_test), y_test)
        self.save_model()
        return scores, accuracy
    
    def predict(self):
        raw = self.preprocess_data()
        X, y = self.get_data(raw)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        preds = []

        print("epoch nb: [prediction] [truth] equal?")
        for i in range(len(X)):
            Xi = X[i][np.newaxis, ...]
            pred = self.model.predict(Xi)
            print(f"epoch {i}:  {pred}    [{y[i]}] {pred[0] == y[i]}")
            preds.append(pred)
            time.sleep(0.5)
        print(f"Accuracy: {accuracy_score(preds, y)}")

    def save_model(self):
        if os.path.isfile('models'):
            os.remove('models')
        if not os.path.exists('models'):
            os.mkdir('models')
        pickle.dump(self.model, open(f"models/S{self.subject}{self.task}", 'wb'))