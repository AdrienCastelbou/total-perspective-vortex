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
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['Fp1.', 'Af7.'], threshold=ica_z_thresh)
    ica.exclude = eog_indices
    ica.plot_scores(eog_scores)
    return ica

def preprocess_data(raw):
    fICE_raw = filter_raw(raw=raw, lo_cut=1, hi_cut=30)
    ica = process_ICA(fICE_raw)

def erp_epochs_segmentation(raw):
    events, event_dict = mne.events_from_annotations(raw)
    print(event_dict)
    event_mapping = {"Rest": 1, "LeftFist-BothFists": 2, "RightFist-BothFeet": 3}
    ig, ax = plt.subplots(figsize=[15, 5])
    mne.viz.plot_events(events, raw.info['sfreq'], event_id=event_mapping, axes=ax)
    plt.show()

def main():
    raw = get_raw_eeg()
    pipeline = EEG_Pipeline(raw=raw)
    pipeline.preprocess()
    erp_epochs_segmentation(raw)
    f_raw = filter_raw(raw)
    erp_epochs_segmentation(f_raw)
    #preprocess_data(raw)



if __name__ == "__main__":
    main()