import mne

class Preprocess:
    def __init__(self, vizualize=False) -> None:
        self.vizualize_ = vizualize

    def setup_channels(self, raw):
        chNames = raw.info["ch_names"]
        chMapping = {ch: ch.split('.')[0] for ch in chNames}
        raw.rename_channels(chMapping)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False)

    def run(self, filename):
        raw = mne.io.read_raw_edf(filename, preload=True)
        self.setup_channels(raw)
        rawFilt = raw.copy().filter(0.1, 30)
        if self.vizualize_ == True:
            raw.compute_psd(fmax=5).plot()
            rawFilt.compute_psd(fmax=5).plot()
            raw.plot()
            rawFilt.plot(block=True)
        return rawFilt
