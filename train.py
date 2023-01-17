import mne
from EEG_Pipeline import *
import sys
import os 
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
import numpy as np
from mne.decoding import CSP
from scipy import linealg

def load_raw_eeg():
    raws = []
    names = []
    if len(sys.argv) < 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 main.py path/to/file.edf path/to/directory")
    for fpath in sys.argv[1:]:
        if os.path.isfile(fpath):
            raws.append(mne.io.read_raw_edf(fpath, preload=True))
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

def get_epochs(raw, tmin=-1, tmax=4):
    events, event_dict = mne.events_from_annotations(raw)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    return epochs

def preprocess_pipeline(raws, tmin=-1, tmax=4):
    epochs_list = []
    for raw in raws:
        setup_channels(raw)
        f_raw = filter_raw(raw, 7., 30.)
        epochs_list.append(get_epochs(f_raw, tmin, tmax))
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

class CSP():
    def __init__(self, n_components=4) -> None:
        self.classes_ = None
        self.filters_ = None
        self.n_components_ = n_components
    

    def _decompose_covs(self, covs, weights):
        n_classes = len(self.classes_)
        if n_classes == 2:
            return linealg.eigh(covs[0], covs[0] + covs[1])
        return linealg.eigh(covs[0], covs[1])
    
    def _compute_covs(self, X, y):
        _, n_channels, _ = X.shape
        covs = []
        weights = []
        for curr_class in self.classes_:
            x_class = X[y == curr_class]
            x_class = np.transpose(x_class, [1, 0, 2]).reshape(n_channels, -1)
            covs.append(np.cov(x_class))
            weights.append(x_class.shape[0])
        return covs, weights
    
    def _order_components(self, eigvals, eigvecs):
        n_classes = len(self.classes_)
        if n_classes == 2:
            idx = np.argsort(np.abs(eigvals - 0.5))[::-1]


    def fit(self, X, y):
        n_epochs, n_channels, n_times = X.shape
        self.classes_ = np.unique(y)
        covs, weights = self._compute_covs(X, y)
        eigvals, eigvecs = self._decompose_covs(covs, weights)
        idx = self._order_components(eigvals, eigvecs)
        return self
    
        

    def transform(self, X):
        return self.filters_ @ X

    def transform_fit(self, X, y):
        return self.fit(X, y).transform(X)

def main():
    raws, names = load_raw_eeg()
    epochs = preprocess_pipeline(raws)
    csp_ = CSP()
    csp_.fit(epochs.get_data(), epochs.events[:,-1])



if __name__ == "__main__":
    main()