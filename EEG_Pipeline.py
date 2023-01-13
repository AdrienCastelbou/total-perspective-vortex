import mne
import matplotlib.pyplot as plt
from autoreject import get_rejection_threshold
import numpy as np

class EEG_Pipeline():
    def __init__(self, raw, name, tstep=1., random_state=42, ica_n_components=.99, ica_z_thresh=1.96, display=["all"], filtered_save="filtered", epochs_save="epochs"):
        self.raw = raw
        self.name = name
        self.f_raw = None
        self.ica = None
        self.reject = None
        self.tstep = tstep
        self.random_state = random_state
        self.ica_n_components = ica_n_components
        self.ica_z_thresh = ica_z_thresh
        self.display = display
        self.filter_save = filtered_save
        self.epochs_save = epochs_save
        self.setup_channels()
 
    
    def setup_channels(self):
        chs = self.raw.info['ch_names']
        chs_mapping = {ch: ch.split(".")[0] for ch in chs }
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


    def process_ICA(self, ica_raw):
        events_ica = mne.make_fixed_length_events(ica_raw, duration=self.tstep)
        epochs_ica = mne.Epochs(ica_raw, events_ica, tmin=0.0, tmax=self.tstep, baseline=None, preload=True)
        self.reject = get_rejection_threshold(epochs_ica)
        self.ica = mne.preprocessing.ICA(n_components=self.ica_n_components, random_state=self.random_state)
        self.ica.fit(epochs_ica, reject=self.reject, tstep=self.tstep)
        eog_indices, eog_scores = self.ica.find_bads_eog(ica_raw, ch_name=['Fp1', 'AF7'], threshold=self.ica_z_thresh)
        self.ica.exclude = eog_indices
        if any(display_param in ["ica", "all"] for display_param in self.display):
            self.ica.plot_components()
            self.ica.plot_properties(epochs_ica, picks=range(0, self.ica.n_components_), psd_args={'fmax': 30})
            self.ica.plot_scores(eog_scores)
            plt.show()
        return self.ica
    

    def erp_epochs_segmentation(self, tmin=-.200, tmax=1.000, baseline=(None, 0)):
        events, event_dict = mne.events_from_annotations(self.f_raw)
        times = np.arange(0, tmax, 0.1)
        epochs = mne.Epochs(self.f_raw, events, event_dict, tmin, tmax, baseline=baseline, preload=True)
        if any(display_param in ["epochs", "all"] for display_param in self.display):
            epochs.average().plot(spatial_colors=True)
            epochs.average().plot_topomap(times=times, average=0.050)
        epochs_post_ica = self.ica.apply(epochs.copy())
        if any(display_param in ["epochs", "all"] for display_param in self.display):
            epochs_post_ica.average().plot(spatial_colors=True)
            epochs_post_ica.average().plot_topomap(times=times, average=0.050)
        return epochs_post_ica


    def preprocess(self):
        if any(display_param in ["raw", "all"] for display_param in self.display):
            self.raw.plot(block=True)
            self.show_psd(self.raw)
        f_raw = self.filter_raw()
        self.f_raw = f_raw
        if any(display_param in ["raw", "all"] for display_param in self.display):
            self.f_raw.plot(block=True)
            self.show_psd(self.f_raw)
        ica_raw = self.filter_raw(lo_cut=1, hi_cut=30)
        ica = self.process_ICA(ica_raw)
        epochs = self.erp_epochs_segmentation()
        epochs.save(f"{self.epochs_save}/{self.name}-epo.fif", overwrite=True)
        self.f_raw.save(f"{self.filter_save}/{self.name}-filt.fif", overwrite=True)
        return epochs