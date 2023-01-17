import mne
import matplotlib.pyplot as plt
from autoreject import get_rejection_threshold
import numpy as np

class EEG_Pipeline():
    def __init__(self, raw, name, random_state=42, display=["all"], filtered_save="filtered", epochs_save="epochs"):
        self.raw = raw.copy()
        self.name = name
        self.f_raw = None
        self.reject = None
        self.random_state = random_state
        self.display = display
        self.filter_save = filtered_save
        self.epochs_save = epochs_save
        self.setup_channels()
 
    
    def setup_channels(self):
        ch_names = self.raw.info['ch_names']
        chs_mapping = {ch: ch.split(".")[0] for ch in ch_names }
        mne.rename_channels(self.raw.info, chs_mapping)
        mne.datasets.eegbci.standardize(self.raw)
        self.raw.set_montage('standard_1020')
        if any(display_param in ["montage", "all"] for display_param in self.display):
            self.raw.plot_sensors(show_names=True)
            plt.show()
    
    def filter_raw(self, lo_cut=0.1, hi_cut=30):
        f_raw = self.raw.copy().filter(lo_cut, hi_cut)
        return f_raw
    

    def show_psd(self, raw, fmax=80):
        spectrum = raw.compute_psd(fmax=fmax)
        spectrum.plot()
        plt.show() 
    

    def erp_epochs_segmentation(self, tmin=-.200, tmax=1.000, baseline=(None, 0)):
        events, event_dict = mne.events_from_annotations(self.f_raw)
        times = np.arange(0, tmax, 0.1)
        epochs = mne.Epochs(self.f_raw, events, event_dict, tmin, tmax, baseline=baseline, preload=True)
        if any(display_param in ["epochs", "all"] for display_param in self.display):
            epochs.average().plot(spatial_colors=True)
            epochs.average().plot_topomap(times=times, average=0.050)


    def preprocess(self):
        if any(display_param in ["raw", "all"] for display_param in self.display):
            self.raw.plot(block=True)
            self.show_psd(self.raw)
        f_raw = self.filter_raw()
        self.f_raw = f_raw
        if any(display_param in ["raw", "all"] for display_param in self.display):
            self.f_raw.plot(block=True)
            self.show_psd(self.f_raw)
        epochs = self.erp_epochs_segmentation()
        epochs.save(f"{self.epochs_save}/{self.name}-epo.fif", overwrite=True)
        self.f_raw.save(f"{self.filter_save}/{self.name}-filt.fif", overwrite=True)
        return epochs


def setup_channels(raw):
    ch_names = raw.info['ch_names']
    chs_mapping = {ch: ch.split(".")[0] for ch in ch_names }
    mne.rename_channels(raw.info, chs_mapping)
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage('standard_1020')
    return raw

def filter_raw(raw, lo_cut=0.1, hi_cut=40):
    return raw.copy().filter(lo_cut, hi_cut)

def preprocess_pipeline(raw, tmin=-1, tmax=4):
    setup_channels(raw)
    f_raw = filter_raw(raw, 7., 30.)
    events, event_dict = mne.events_from_annotations(raw)
    print(event_dict)
    picks = mne.pick_types(f_raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(f_raw, events, event_dict, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1]
    print(labels)