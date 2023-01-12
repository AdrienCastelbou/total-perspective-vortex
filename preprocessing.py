import mne
import sys
import matplotlib.pyplot as plt
from EEG_Pipeline import EEG_Pipeline
import os


def load_datas():
    raws = []
    names = []
    if len(sys.argv) != 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 main.py path/to/file.edf (can be a directory)")
    path = sys.argv[1]
    if os.path.isfile(path):
        raws.append(mne.io.read_raw_edf(path, preload=True))
        names.append(os.path.basename(path))
    else:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for f in files:
            raws.append(mne.io.read_raw_edf(path + "/" + f, preload=True))
            names.append(f)
    return raws, names


def show_psd(raw, fmax=80):
    spectrum = raw.compute_psd(fmax=fmax)
    spectrum.plot()
    plt.show() 


def main():
    raws, names = load_datas()
    for raw, name in zip(raws, names):
        pipeline = EEG_Pipeline(raw, name, display=None)
        pipeline.preprocess()


if __name__ == "__main__":
    main()