import mne
import matplotlib.pyplot as plt
from autoreject import get_rejection_threshold

class EEG_Pipeline():
    def __init__(self, raw, tstep=1., random_state=42, ica_n_components=.99, ica_z_thresh=1.96):
        self.raw = raw
        self.f_raw = None
        self.ica = None
        self.reject = None
        self.tstep = tstep
        self.random_state = random_state
        self.ica_n_components = ica_n_components
        self.ica_z_thresh = ica_z_thresh
    

    def filter_raw(self, lo_cut=0.1, hi_cut=30):
        f_raw = self.raw.copy().filter(lo_cut, hi_cut)
        return f_raw
    

    def process_ICA(self, ica_raw):
        events_ica = mne.make_fixed_length_events(ica_raw, duration=self.tstep)
        epochs_ica = mne.Epochs(ica_raw, events_ica, tmin=0.0, tmax=self.tstep, baseline=None, preload=True)
        self.reject = get_rejection_threshold(epochs_ica)
        self.ica = mne.preprocessing.ICA(n_components=self.ica_n_components, random_state=self.random_state)
        self.ica.fit(epochs_ica, reject=self.reject, tstep=self.tstep)
        eog_indices, eog_scores = self.ica.find_bads_eog(ica_raw, ch_name=['Fp1.', 'Af7.'], threshold=self.ica_z_thresh)
        self.ica.exclude = eog_indices
        self.ica.plot_scores(eog_scores)
        return self.ica
    

    def preprocess(self):
        f_raw = self.filter_raw()
        self.f_raw = f_raw
        ica_raw = self.filter_raw(lo_cut=1, hi_cut=30)
        ica = self.process_ICA(ica_raw)