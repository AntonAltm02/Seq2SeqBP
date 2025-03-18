import os
import numpy as np
import pandas
import mat73
import scipy.io
from scipy.stats import zscore
import pywt
import neurokit2 as nk
from scipy.signal import cheby2, butter, filtfilt
from scipy.signal import find_peaks, correlate


class PreprocessingPulseDB:
    def __init__(self, data_path, target_path, f_low, f_high, order, fs=125, db='m', replace=False):
        ##
        # @brief This constructor initializes the DataGenerator object.
        # @param data_path      The main path which includes the data of the mimic3 or the vitaldb with all subjects.
        # @param target_path    The target path, where the preprocessed data need to be saved.
        # @param fs             The target sampling rate.
        # @param old_fs         The sampling rate of the raw database.
        # @param new_nr_sample  The new number of sample per time epoch
        # @param db             This parameter set the database, which should be preprocessed. 'm'(Default) for Mimic3 and 'v' for VitalDB.
        ##
        self.data_path = data_path
        self.target_path = target_path
        self.fs = fs
        self.db = db

        self.order = order
        self.Wn = [f_low / (self.fs / 2), f_high / (self.fs / 2)]

        # skips every file which is already present in the directory
        self.ids = os.listdir(self.data_path)
        if not replace:
            id_ready = os.listdir(self.target_path + "data/")
            self.ids = [x for x in self.ids if x[:-4] + self.db + '.npy' not in id_ready]

        self.keys = None

        self._ppg_segments = None
        self._abp_segments = None
        self._features = None
        self._dbp = None
        self._sbp = None

        self._sos = None
        self._mean = None
        self._var = None
        self.data_error = False
        self._no_error = True

    def _error_handling(self, _id):
        if self._no_error:
            with open(self.target_path + self.db + '_error.txt', 'w') as file:
                file.write('Following ID(s) have caused an error:\n')
                file.write(_id[:-4] + self.db + "\n")
        else:
            with open(self.target_path + self.db + '_error.txt', 'a') as file:
                file.write(_id[:-4] + self.db + "\n")
        print(_id, " creates an error!")
        self.data_error = True

    def _load_data(self, _id):
        ##
        # @brief This method loads data of the next subject.
        ##   
        # Loading MIMIC3
        if self.db == 'm':
            try:
                try:
                    data_dict = mat73.loadmat(self.data_path + _id)
                    data = data_dict['Subj_Wins']
                    if self.keys is None:
                        keys = data.keys()
                        self.keys = [x for x in keys]
                    self._sbp = data["SegSBP"]
                    self._dbp = data["SegDBP"]
                    if type(self._sbp) == list:
                        self._ppg_segments = np.squeeze(np.float32(data['PPG_Raw']), axis=1)
                        self._abp_segments = np.squeeze(np.float32(data["ABP_Raw"]), axis=1)
                        self._sbp = np.squeeze(np.float32(data["SegSBP"]), axis=1)
                        self._dbp = np.squeeze(np.float32(data["SegDBP"]), axis=1)
                    else:
                        self._ppg_segments = np.float32(np.expand_dims(data['PPG_Raw'], axis=0))
                        self._abp_segments = np.float32(np.expand_dims(data["ABP_Raw"], axis=0))
                        self._sbp = np.float32(np.expand_dims(data["SegSBP"], axis=0))
                        self._dbp = np.float32(np.expand_dims(data["SegDBP"], axis=0))

                except:
                    data_dict = scipy.io.loadmat(self.data_path + _id)
                    data = data_dict['Subj_Wins'][0][0]
                    self._ppg_segments = np.float32(np.swapaxes(data[9], 0, 1))

            except:
                self._error_handling(_id)

        # Loading VitalDB
        elif self.db == 'v':
            try:
                try:
                    data_dict = mat73.loadmat(self.data_path + _id)
                    data = data_dict['Subj_Wins']
                    if self.keys is None:
                        keys = data.keys()
                        self.keys = [x for x in keys]
                    self._ppg_segments = np.squeeze(np.float32(data['PPG_Raw']), axis=1)
                    self._abp_segments = np.squeeze(np.float32(data["ABP_Raw"]), axis=1)
                    self._sbp = np.squeeze(np.float32(data["SegSBP"]), axis=1)
                    self._dbp = np.squeeze(np.float32(data["SegDBP"]), axis=1)
                except:
                    data_dict = mat73.loadmat(self.data_path + _id)
                    data = data_dict['Subj_Wins']
                    if self.keys is None:
                        keys = data.keys()
                        self.keys = [x for x in keys]
                    # self._ppg_segments = np.swapaxes(np.float32(data['PPG_Raw']), 0, 1)
                    self._ppg_segments = np.float32(np.expand_dims(data['PPG_Raw'], axis=0))
                    self._abp_segments = np.float32(np.expand_dims(data["ABP_Raw"], axis=0))
                    self._sbp = np.float32(np.expand_dims(data["SegSBP"], axis=0))
                    self._dbp = np.float32(np.expand_dims(data["SegDBP"], axis=0))
            except:
                self._error_handling(_id)

    def _wavelet(self, level=2, threshold=10):

        coeffs = pywt.wavedec(self._ppg_segments, "sym4", level=level)

        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold, mode="soft")

        self._ppg_segments = pywt.waverec(coeffs=coeffs, wavelet="sym4")

    def _filtering(self):
        # b, a = cheby2(N=self.order, rs=0.5, Wn=self.Wn, btype="bandpass", analog=False, output="ba")
        b, a = butter(N=self.order, Wn=self.Wn, btype="bandpass", analog=False, output="ba")

        self._ppg_segments = filtfilt(b, a, self._ppg_segments)

    def _standardize(self, _id):
        ##
        # @brief This method standardizes the data of one subject.
        ##   
        self._ppg_segments = zscore(self._ppg_segments)

        for i in range(len(self._ppg_segments)):
            min_val = np.min(self._ppg_segments[i])
            max_val = np.max(self._ppg_segments[i])
            self._ppg_segments[i] = (self._ppg_segments[i] - min_val) / (max_val - min_val)

    def extract_features_event_related(self):
        signal = np.concatenate(self._ppg_segments)
        signal_clean = nk.ppg_process(ppg_signal=signal, sampling_rate=self.fs)
        epochs = nk.epochs_create(signal_clean, sampling_rate=self.fs, epochs_start=0, epochs_end=10)
        self._features = nk.ppg_analyze(data=epochs, sampling_rate=self.fs, method="event-related")

    def extract_features_interval_related(self):
        self._features = pandas.DataFrame()

        for i in range(len(self._ppg_segments)):
            signals, _ = nk.ppg_process(ppg_signal=self._ppg_segments[i], sampling_rate=200)
            new_features = nk.ppg_analyze(data=signals, sampling_rate=self.fs, method="interval-related")
            self._features = pandas.concat([self._features, new_features], ignore_index=True)

    def _phase_align(self, _id):
        """
        Phase align the PPG and BP signals aligned_ppg
        aligned_bp = phase_align(ppg_signal, bp_signal)
        """

        # Find the peak indices of the PPG and BP signals
        aligned_ppg = np.zeros(self._ppg_segments.shape)
        aligned_bp = np.zeros(self._abp_segments.shape)
        for i in range(len(self._ppg_segments)):
            ppg_peaks, _ = find_peaks(self._ppg_segments[i])
            bp_peaks, _ = find_peaks(self._abp_segments[i])

            # Compute the cross-correlation of the PPG and BP signals
            cross_corr = correlate(self._ppg_segments[i], self._abp_segments[i], mode='same')

            # Find the index of the maximum value in the cross-correlation function
            max_index = np.argmax(cross_corr)

            # Determine the phase difference (in samples) between the PPG and BP signals
            ppg_offset = max_index - len(self._abp_segments[i]) // 2

            if np.abs(ppg_offset) > 125:
                continue
            else:
                # Align the PPG and BP signals based on the phase difference
                if ppg_offset > 0:
                    aligned_ppg[i] = np.concatenate([self._ppg_segments[i][ppg_offset:], np.zeros(ppg_offset)])
                    aligned_bp[i] = np.copy(self._abp_segments[i])
                elif ppg_offset < 0:
                    aligned_ppg[i] = np.copy(self._ppg_segments[i])
                    aligned_bp[i] = np.concatenate([self._abp_segments[i][-ppg_offset:], np.zeros(-ppg_offset)])
                else:
                    aligned_ppg[i] = np.copy(self._ppg_segments[i])
                    aligned_bp[i] = np.copy(self._abp_segments[i])

        ppg_non_zero_rows = ~np.all(aligned_ppg == 0, axis=1)
        abp_non_zero_rows = ~np.all(aligned_bp == 0, axis=1)

        aligned_ppg = aligned_ppg[ppg_non_zero_rows]
        aligned_bp = aligned_bp[abp_non_zero_rows]
        self._sbp = self._sbp[0:len(aligned_ppg)]
        self._dbp = self._dbp[0:len(aligned_ppg)]

        return aligned_ppg, aligned_bp

    def _decrease_length(self, decrease_len):
        decrease_ppg = np.zeros((len(self._ppg_segments), decrease_len))
        decrease_abp = np.zeros((len(self._abp_segments), decrease_len))
        for i in range(len(self._ppg_segments)):
            decrease_ppg[i] = self._ppg_segments[i, :decrease_len]
            decrease_abp[i] = self._abp_segments[i, :decrease_len]

        self._ppg_segments = decrease_ppg
        self._abp_segments = decrease_abp

    def process(self):
        ##
        # @brief This method summarize all preprocessing steps in form of a Pipeline
        ##
        num_segments = 0

        for nr_sub, sub_id in enumerate(self.ids):
            print("Load Subject ", str(nr_sub + 1), " of ", str(len(self.ids)))
            # Load Data
            self._load_data(sub_id)
            if not self.data_error:

                # filtering
                self._filtering()

                # wavelet transformation
                self._wavelet()

                # Phase matching
                self._ppg_segments, self._abp_segments = self._phase_align(sub_id)

                # Standardize
                self._standardize(sub_id)

                # Decrease the length to 8-s (1000 samples)
                self._decrease_length(decrease_len=1000)

                # Check if segments contain NaN values
                nan_rows_ppg = np.isnan(self._ppg_segments).any(axis=1)
                nan_rows_abp = np.isnan(self._abp_segments).any(axis=1)
                self._ppg_segments = self._abp_segments[~nan_rows_ppg]
                self._abp_segments = self._abp_segments[~nan_rows_abp]

                if not np.all(self._ppg_segments == 0) and not np.all(self._abp_segments == 0) and not np.all(
                        self._sbp == 0) and not np.all(self._dbp == 0):
                    # Save Data
                    np.save(self.target_path + 'data/' + str(sub_id[:-4]) + self.db, self._ppg_segments)
                    np.save(self.target_path + 'ground_truth/' + str(sub_id[:-4]) + self.db, self._abp_segments)
                    np.save(self.target_path + 'sbp/' + str(sub_id[:-4]) + self.db, self._sbp)
                    np.save(self.target_path + 'dbp/' + str(sub_id[:-4]) + self.db, self._dbp)
                    num_segments += len(self._ppg_segments)
                else:
                    print("The matrix contains only zeros. Skipping the saving process.")

            else:
                self.data_error = False

        print(f"Total number of segments: {num_segments}")
