import mne
import sys
import matplotlib.pyplot as plt
from autoreject import get_rejection_threshold
from EEG_Pipeline import EEG_Pipeline


def get_raw_eeg():
    if len(sys.argv) != 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 main.py name_of_remote_eeg")
    raw = mne.io.read_raw_edf(sys.argv[1], preload=True)
    return raw


def show_psd(raw, fmax=80):
    spectrum = raw.compute_psd(fmax=fmax)
    spectrum.plot()
    plt.show() 


def main():
    raw = get_raw_eeg()
    pipeline = EEG_Pipeline(raw=raw)
    pipeline.preprocess()


if __name__ == "__main__":
    main()