import os
from sklearn.model_selection import train_test_split
from Implementierung.Data.data_generator import DataGenerator
import scipy
import neurokit2
import numpy as np


class Mean_regressor:
    def __init__(self):
        self.path_main = "D:/PulseDB/Data_proc/"
        self.batch_size = 32
        files = os.listdir(self.path_main + "data/")
        train_id, self.test_id = train_test_split(files, test_size=0.3, random_state=42)
        self.train_id, self.val_id = train_test_split(train_id, test_size=0.3, random_state=42)

        # filter
        fs = 125
        nyq = 0.5 * fs
        low_freq = 0.1 / nyq
        high_freq = 5 / nyq
        self.Wn = [low_freq, high_freq]
        self.b, self.a = scipy.signal.cheby2(N=4, rs=0.5, Wn=self.Wn, btype="bandpass", analog=False, output="ba")

    def get_peaks(self, signal):
        extract_max, _ = neurokit2.ppg_process(signal, sampling_rate=125)
        idx_max = np.where(extract_max.PPG_Peaks == 1)[0]

        extract_min = []
        for i in range(len(idx_max) - 1):
            start_index = idx_max[i]
            end_index = idx_max[i + 1]
            cycle = signal[start_index:end_index + 1]
            extract_min.append(start_index + np.argmin(cycle))
        idx_min = np.array(extract_min)

        sbp = np.mean(signal[idx_max])
        dbp = np.mean(signal[idx_min])

        return sbp, dbp

    def calc_sbp_dbp(self, train, test):
        train = train.squeeze(-1)
        test = test.squeeze(-1)

        train_sbp = []
        train_dbp = []

        for i in range(len(train)):
            sbp, dbp = self.get_peaks(train[i, :])
            train_sbp.append(sbp), train_dbp.append(dbp)

        mean_train_sbp = np.mean(np.vstack(train_sbp))
        mean_train_dbp = np.mean(np.vstack(train_dbp))

        test_sbp = []
        test_dbp = []
        for i in range(len(test)):
            sbp, dbp = self.get_peaks(test[i, :])
            test_sbp.append(sbp), test_dbp.append(dbp)

        test_sbp = np.vstack(test_sbp)
        test_dbp = np.vstack(test_dbp)

        # Calculate mean error
        error_sbp = mean_train_sbp - test_sbp
        me_sbp = np.mean(error_sbp)
        error_dbp = mean_train_dbp - test_dbp
        me_dbp = np.mean(error_dbp)
        print(f"Mean Error SBP: {me_sbp:.2f}, Mean Error DBP: {me_dbp:.2f}")

        # Calculate mean absolute error
        ae_sbp = np.abs(mean_train_sbp - test_sbp)
        mae_sbp = np.mean(ae_sbp)
        ae_dbp = np.abs(mean_train_dbp - test_dbp)
        mae_dbp = np.mean(ae_dbp)
        print(f"Mean Absolute Error SBP: {mae_sbp:.2f}, Mean Absolute Error DBP: {mae_dbp:.2f}")

        # Calculate standard deviation of errors
        std_sbp = np.std(error_sbp)
        std_dbp = np.std(error_dbp)
        print(f"Standard Deviation SBP: {std_sbp:.2f}, Standard Deviation DBP: {std_dbp:.2f}")

    def main(self):
        train_generator = DataGenerator(path_main=self.path_main, list_id=self.train_id, batch_size=self.batch_size,
                                        shuffle=True)
        test_generator = DataGenerator(path_main=self.path_main, list_id=self.test_id, batch_size=self.batch_size,
                                       shuffle=True)
        abp_train = []
        abp_test = []
        for _, abp, _, _ in train_generator:
            abp_train.append(abp)
        for _, abp, _, _ in test_generator:
            abp_test.append(abp)

        abp_train = np.concatenate(abp_train)
        abp_test = np.concatenate(abp_test)
        self.calc_sbp_dbp(abp_train, abp_test)


if __name__ == "__main__":
    mean_regressor = Mean_regressor()
    mean_regressor.main()
