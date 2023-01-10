import mne
import sys
import matplotlib.pyplot as plt

def get_raw_eeg():
    if len(sys.argv) != 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 main.py name_of_remote_eeg")
    raw = mne.io.read_raw_edf(sys.argv[1], preload=True)
    return raw


def show_psd(raw, fmax=80):
    spectrum = raw.compute_psd(fmax=fmax)
    spectrum.plot()
    plt.show() 


def filter_raw(raw, lo_cut=0.1, hi_cut=30):
    filtered_raw = raw.copy().filter(lo_cut, hi_cut)
    return filtered_raw


def main():
    raw = get_raw_eeg()
    raw.plot(block=True, start=15, duration=5)
    f_raw = filter_raw(raw)
    f_raw.plot(block=True, start=15, duration=5)



if __name__ == "__main__":
    main()