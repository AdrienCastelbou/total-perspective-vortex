import mne

class Preprocess:
    def __init__(self, vizualize=False, annotations=None) -> None:
        self.vizualize_ = vizualize
        self.annotations = annotations

        
    def setup_channels(self, raw):
        chNames = raw.info["ch_names"]
        chMapping = {ch: ch.split('.')[0] for ch in chNames}
        raw.rename_channels(chMapping)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False)

    def run(self, filenames):
        raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in filenames])
        self.setup_channels(raw)
        if (self.annotations):
            raw.annotations.rename(self.annotations)
        rawFilt = raw.copy().filter(7, 30)
        if self.vizualize_ == True:
            raw.compute_psd(fmax=50).plot()
            rawFilt.compute_psd(fmax=50).plot()
            raw.plot()
            rawFilt.plot(block=True)
        return rawFilt
