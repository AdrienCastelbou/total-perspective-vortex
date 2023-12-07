import sys
import os.path
from enums.process import Process
from process.Preprocess import Preprocess
from utils.experiments import experiments
import mne
from mne.decoding import CSP
import numpy as np
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def preprocess_data(filenames, vizualize=False):
    preprocessor = Preprocess(vizualize)
    raw = preprocessor.run(filenames)
    return raw
    

def train(filenames):
    tmin, tmax = -1.0, 4.0

    raw = preprocess_data(filenames)
    events, _ = mne.events_from_annotations(raw)
    picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, exclude="bads")
    epochs = mne.Epochs(raw, events,tmin=tmin, tmax=tmax, picks=picks, proj=True, baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
    labels = epochs.events[:, -1]

    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)


    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)
    
    print(np.mean(scores))
    return np.mean(scores)
    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)

    #csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)").savefig("CSP Patterns.png")
    #csp.plot_filters(epochs.info, ch_type='eeg').savefig("CSP Filters.png")
    
    sfreq = raw.info["sfreq"]
    w_length = int(sfreq * 0.5)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
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
            X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)
    # Plot scores over time
    w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
    plt.axvline(0, linestyle="--", color="k", label="Onset")
    plt.axhline(0.5, linestyle="-", color="k", label="Chance")
    plt.xlabel("time (s)")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")
    plt.legend(loc="lower right")
    plt.show()
    return
    raw_data = raw.get_data()
    print(raw_data.shape)
    events, event_dict = mne.events_from_annotations(raw)
    print(event_dict)
    epochs = mne.Epochs(raw, events, tmax=2)
    rest_epochs = epochs['1'].get_data()
    first_action_epochs = epochs['2'].get_data()
    scnd_action_epochs = epochs['3'].get_data()
    print(epochs, rest_epochs.shape, first_action_epochs.shape, scnd_action_epochs.shape)
    cov_matrix_rest = np.mean([np.cov(epoch, rowvar=False) for epoch in rest_epochs], axis = 0)
    cov_matrix_first = np.mean([np.cov(epoch, rowvar=False) for epoch in first_action_epochs], axis = 0)
    cov_matrix_scnd = np.mean([np.cov(epoch, rowvar=False) for epoch in scnd_action_epochs], axis = 0)
    print(cov_matrix_rest.shape, cov_matrix_first.shape, cov_matrix_scnd.shape)
    eigenvalues_first, eigenvectors_first = np.linalg.eigh(cov_matrix_first, cov_matrix_first,UPLO='L')
    return
    eigenvalues_scnd, eigenvectors_scnd = np.linalg.eigh(cov_matrix_scnd, cov_matrix_scnd + cov_matrix_rest)
    filtered_first_data = np.dot(raw, eigenvectors_first[:, :2])
    filtered_scnd_data = np.dot(raw, eigenvectors_scnd[:, :2])
    print(filtered_first_data.shape)



def get_paths(subject, run, getAllRuns = False):
    filenames = []
    runs = []
    for task in experiments:
        if run in experiments[task]:
            if getAllRuns:
                runs = experiments[task]
            else:
                runs.append(run)
            break
    if runs == []:
        raise Exception(f"Run {run} of subject {subject} is not valable, please use a number between 3 and 14")
    subjectId = subject.zfill(3)
    for r in runs:
        runId = r.zfill(2)
        filename = f"./physionet.org/files/eegmmidb/1.0.0/S{subjectId}/S{subjectId}R{runId}.edf"
        if (not os.path.isfile(filename)):
            raise Exception(f"Run {run} of subject {subject} does not exist, make sure the corresponding file exists in ./physionet.org/files/eegmmidb/1.0.0/SXXX/SXXXRXX.edf")
        filenames.append(filename)
    return filenames

def get_params():
    args = sys.argv[1:]

    if len(args) == 0:
        return Process.ALL, None
    elif len(args) != 3 or args[2] not in Process:
        raise Exception("Error: wrong usage -> python mybci.py [subject id (int)] [experiment id (int)] [process (preprocess/train/predict)]\nExample : python mybci.py 1 14 train")
    filenames = get_paths(args[0], args[1], args[2] != Process.PREPROCESS)
    
    return args[2], filenames


def train_all_models():
    means = []
    for experiment in range(3, 7):
        experiment_means = []
        for subject in range(1, 36):
            filenames = get_paths(str(subject), str(experiment))
            mean = train(filenames)
            print(f"Subject {subject} experiment {experiment} : {mean}")
            experiment_means.append(mean)
        experiment_mean = experiment_means
        means.append(experiment_mean)
        print(f"experiment {experiment}:    accuracy = {np.mean(experiment_mean)}")
        print('-----------------------------------')
    print(f"Mean accuracy of 6 experiments: {np.mean(means)}")

def main():
    try:
        mne.set_log_level('WARNING')
        process, filenames = get_params()
        if process == Process.PREPROCESS:
           preprocess_data(filenames, True)
        elif process == Process.TRAIN:
            train(filenames)
        elif process == Process.ALL:
            train_all_models()
    except Exception as e:
        print(e)
        return
    

if __name__ == "__main__":
    main()