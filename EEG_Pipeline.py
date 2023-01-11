import mne
import matplotlib.pyplot as plt
from autoreject import get_rejection_threshold

class EEG_Pipeline():
    def __init__(self, raw, name, tstep=1., random_state=42, ica_n_components=.99, ica_z_thresh=1.96):
        self.raw = raw
        self.name = name
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
        return self.ica
    

    def erp_epochs_segmentation(self, tmin=-.200, tmax=1.000, baseline=(None, 0)):
        events, event_dict = mne.events_from_annotations(self.f_raw)
        event_mapping = {"Rest": 1, "Match/LF-Fists": 2, "Match/RF-BFeet": 3}
        epochs = mne.Epochs(self.f_raw, events, event_mapping, tmin, tmax, baseline=baseline, preload=True)
        epochs.average().plot(spatial_colors=True)
        epochs_post_ica = self.ica.apply(epochs.copy())
        epochs_post_ica.average().plot(spatial_colors=True)
        return epochs_post_ica


    def preprocess(self):
        f_raw = self.filter_raw()
        self.f_raw = f_raw
        ica_raw = self.filter_raw(lo_cut=1, hi_cut=30)
        ica = self.process_ICA(ica_raw)
        epochs = self.erp_epochs_segmentation()
        epochs.save(f"{self.name}-epo.fif")
        return epochs