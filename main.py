import mne
import sys
import matplotlib.pyplot as plt
from autoreject import get_rejection_threshold

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

def segment_data_for_ICE(raw, tstep=1.):
    events_ica = mne.make_fixed_length_events(raw, duration=tstep)
    epochs_ica = mne.Epochs(raw, events_ica, tmin=0.0, tmax=tstep, baseline=None, preload=True)
    return epochs_ica


def process_ICA(raw, tstep=1., random_state=42, ica_n_components=.99):
    epochs_ica = segment_data_for_ICE(raw, tstep)
    reject = get_rejection_threshold(epochs_ica)
    ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state)
    ica.fit(epochs_ica, reject=reject, tstep=tstep)
    ica_z_thresh = 1.96
    print(raw.ch_names)
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['Fp1.', 'Af7.'], threshold=ica_z_thresh)
    ica.exclude = eog_indices
    ica.plot_scores(eog_scores)

def preprocess_data(raw):
    fICE_raw = filter_raw(raw=raw, lo_cut=1, hi_cut=30)
    process_ICA(fICE_raw)

def main():
    raw = get_raw_eeg()
    f_raw = filter_raw(raw)
    preprocess_data(raw)



if __name__ == "__main__":
    main()