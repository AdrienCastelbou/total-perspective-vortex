import mne
import sys
import matplotlib.pyplot as plt
from EEG_Pipeline import EEG_Pipeline


def get_eeg():
    if len(sys.argv) != 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 main.py name_of_remote_eeg")
    filename = sys.argv[1]
    raw = mne.io.read_raw_edf(filename, preload=True)
    return raw, filename[:-4]


def show_psd(raw, fmax=80):
    spectrum = raw.compute_psd(fmax=fmax)
    spectrum.plot()
    plt.show() 


def main():
    raw, name = get_eeg()
    pipeline = EEG_Pipeline(raw, name)
    pipeline.preprocess()


if __name__ == "__main__":
    main()