import sys
import os.path
from enums.process import Process
from process.Preprocess import Preprocess
from utils.experiments import experiments
import mne
from mne.decoding import UnsupervisedSpatialFilter
from mne.preprocessing import ICA
from mne.io import concatenate_raws, read_raw_edf
from sklearn.decomposition import PCA
import numpy as np

def preprocess_data(filenames, vizualize=False):
    preprocessor = Preprocess(True)
    raw = preprocessor.run(filenames)
    return raw
    

def train(filename):
    raw = preprocess_data(filename)
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
        return "all", None
    elif len(args) != 3 or args[2] not in Process:
        raise Exception("Error: wrong usage -> python mybci.py [subject id (int)] [experiment id (int)] [process (preprocess/train/predict)]\nExample : python mybci.py 1 14 train")
    filenames = get_paths(args[0], args[1], args[2] != Process.PREPROCESS)
    
    return args[2], filenames

def main():
    try:
        process, filenames = get_params()
        print(filenames)
    except Exception as e:
        print(e)
        return
    if process == Process.PREPROCESS:
        preprocess_data(filenames, True)
    elif process == Process.TRAIN:
        train(filenames)
    

if __name__ == "__main__":
    main()