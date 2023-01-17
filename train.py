import mne
from EEG_Pipeline import *
import sys
import os 
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
import numpy as np
from mne.decoding import CSP
from const import *

def load_raw_eeg():
    raws = []
    names = []
    if len(sys.argv) < 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 main.py path/to/file.edf path/to/directory")
    for fpath in sys.argv[1:]:
        if os.path.isfile(fpath):
            raws.append(mne.io.read_raw_edf(fpath))
            names.append(os.path.basename(fpath))
        else:
            files = [f for f in os.listdir(fpath) if os.path.isfile(os.path.join(fpath, f))]
            for f in files:
                raws.append(mne.io.read_raw_edf(fpath + "/" + f, preload=True, verbose=False))
                names.append(f)
    return raws, names


def setup_channels(raw):
    ch_names = raw.info['ch_names']
    chs_mapping = {ch: ch.split(".")[0] for ch in ch_names }
    mne.rename_channels(raw.info, chs_mapping)
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage('standard_1020')
    return raw

def filter_raw(raw, lo_cut=0.1, hi_cut=40):
    return raw.copy().filter(lo_cut, hi_cut)

def get_epochs(raw, dict, annot_dict, tmin=-1, tmax=4):
    events, _ = mne.events_from_annotations(raw, annot_dict)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(raw, events, dict, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    return epochs

def get_event_dictionnary(name):
    n_experimental_run = int(name[name.find("R") + len("R"): name.find(".edf")])
    if n_experimental_run in [3, 4, 7, 8, 11, 12]:
        return task_one_two_dict, task_one_two_annot_dict
    elif n_experimental_run in [5, 6, 9, 10, 13, 14]:
        return task_three_four_dict, task_three_four_annot_dict
    return rest_dict, rest_annot_dict

def preprocess_pipeline(raws, names, tmin=-1, tmax=4):
    epochs_list = []
    for raw, name in zip(raws, names):
        event_dict, annot_dict = get_event_dictionnary(name)
        setup_channels(raw)
        f_raw = filter_raw(raw, 7., 30.)
        epochs_list.append(get_epochs(f_raw,event_dict, annot_dict, tmin, tmax))
    epochs = mne.concatenate_epochs(epochs_list, True)
    return epochs


def train(epochs):
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs.copy().crop(tmin=1., tmax=2.).get_data()
    labels = epochs.events[:,-1]
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)
    lda = LinearDiscriminantAnalysis()
    csp = mne.decoding.CSP(n_components=4, reg=None, log=True, norm_trace=False)
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                            class_balance))

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)

    csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
    plt.show()
    w_length = int(160 * 0.5)   # running classifier: window length
    w_step = int(160 * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
        X_test = csp.transform(epochs_data_train[test_idx])

        # fit classifier
        lda.fit(X_train, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)

    # Plot scores over time
    w_times = (w_start + w_length / 2.) / 160 + epochs.tmin

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
    plt.axvline(0, linestyle='--', color='k', label='Onset')
    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    plt.xlabel('time (s)')
    plt.ylabel('classification accuracy')
    plt.title('Classification score over time')
    plt.legend(loc='lower right')
    plt.show()


def get_csp(epochs):
    labels = epochs.events[:,-1]
    csp = CSP(n_components=2)
    v = csp.fit_transform(epochs.get_data(), labels)
    print(v, v.shape)
    #t = csp.transform(epochs)
    #print(t, t.shape)
    print(csp.filters_.shape)

def calculate_csp(epochs):
    labels = epochs.events[:,-1]
    print(epochs.get_data().shape)
    epochs_1 = np.transpose(epochs["T1"].get_data(), [1, 0, 2]).reshape(64, -1)
    epochs_2 = np.transpose(epochs["T2"].get_data(), [1, 0, 2]).reshape(64, -1)
    cov_1 = np.cov(epochs_1)
    cov_2 = np.cov(epochs_2)
    eigvals, eigvecs = eig(cov_1, cov_2)
    D_matrix = np.diag(np.sort(eigvals)[::-1])


class ICA():
    def __init__(self) -> None:
        self.classes_ = None
        self.filter_ = None


    def center_(self, X):
        mean = np.mean(X, axis=1, keepdims=True)
        centered =  X - mean
        return centered, mean

    def whiten_(self, X):
        cov = np.cov(X)
        U, S, V = np.linalg.svd(cov)
        d = np.diag(1.0 / np.sqrt(S))
        wM = U @ d @ U.T
        Xw = wM @ X
        print(np.cov(wM))
        return Xw, wM

    def kurtosis(x):
        n = np.shape(x)[0]
        mean = np.sum((x**1)/n) # Calculate the mean
        var = np.sum((x-mean)**2)/n # Calculate the variance
        skew = np.sum((x-mean)**3)/n # Calculate the skewness
        kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
        kurt = kurt/(var**2)-3
        return kurt, skew, var, mean
    
    def fastIca_(self, signals,  alpha = 1, thresh=1e-8, iterations=100):
        m, n = signals.shape
        # Initialize random weights
        print("start")
        W = np.random.rand(m, m)
        for c in range(m):
                w = W[c, :].copy().reshape(m, 1)
                w = w/ np.sqrt((w ** 2).sum())
                i = 0
                lim = 100
                while ((lim > thresh) and (i < iterations)):
                    # Dot product of weight and signal
                    ws = np.dot(w.T, signals)
                    # Pass w*s into contrast function g
                    wg = np.tanh(ws * alpha).T
                    # Pass w*s into g'
                    wg_ = (1 - np.square(np.tanh(ws))) * alpha
                    # Update weights
                    wNew = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()
                    # Decorrelate weights              
                    wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
                    wNew = wNew / np.sqrt((wNew ** 2).sum())
                    # Calculate limit condition
                    lim = np.abs(np.abs((wNew * w).sum()) - 1)
                        
                    # Update weights
                    w = wNew
                        
                    # Update counter
                    i += 1
                W[c, :] = w.T
        print('end')
        return W

    def fit(self, X, y):
        n_epochs, n_channels, n_times = X.shape
        X = np.transpose(X, [1, 0, 2]).reshape(n_channels, -1)
        Xc, meanX = self.center_(X)
        # Whiten mixed signals
        Xw, whiteM = self.whiten_(Xc)
        # Run the ICA to estimate W
        W = self.fastIca_(Xw,  alpha=1)
        #Un-mix signals using W
        unMixed = Xw.T.dot(W.T)
        # Subtract mean from the unmixed signals
        fig, ax = plt.subplots(1, 1, figsize=[18, 10])
        ax.plot(unMixed)
        fig, ax = plt.subplots(1, 1, figsize=[18, 5])
        ax.plot(Xc)        
        plt.show()
    



def main():
    raws, names = load_raw_eeg()
    epochs = preprocess_pipeline(raws, names)
    print(epochs, epochs.events[:,-1])
    train(epochs)



if __name__ == "__main__":
    main()