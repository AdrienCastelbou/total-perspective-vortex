import mne
import sys
import matplotlib.pyplot as plt
from EEG_Pipeline import EEG_Pipeline
import os


def load_datas():
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
                raws.append(mne.io.read_raw_edf(fpath + "/" + f, preload=True))
                names.append(f)
    return raws, names


def show_psd(raw, fmax=80):
    spectrum = raw.compute_psd(fmax=fmax)
    spectrum.plot()
    plt.show() 


def main():
    raws, names = load_datas()
    for raw, name in zip(raws, names):
        chs = raw.info['ch_names']
        chs_mapping = {ch: ch.split(".")[0] for ch in chs }
        mne.rename_channels(raw.info, chs_mapping)
        mne.datasets.eegbci.standardize(raw)
        raw.set_montage('standard_1020')
        raw.plot_sensors(show_names=True)
        plt.show()
        pipeline = EEG_Pipeline(raw, name)
        pipeline.preprocess()


if __name__ == "__main__":
    main()