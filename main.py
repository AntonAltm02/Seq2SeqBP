from Implementierung.preprocessing_PulseDB import PreprocessingPulseDB
from Implementierung.test import Test
from Implementierung.train import Train

data_path_MIMIC = "D:/Download/PulseDB_MIMIC/"
data_path_Vital = "D:/Download/PulseDB_Vital/"
target_path_MIMIC = "D:/Download/MIMIC_proc/"
target_path_Vital = "D:/Download/Vital_proc/"

data_path_train = "D:/Download/PulseDB_proc_train/train_Data/"
data_path_test = "D:/Download/PulseDB_proc_test/test_Data/"
target_path_train = "D:/Download/PulseDB_proc_train/"
target_path_test = "D:/Download/PulseDB_proc_test/"
fs = 125


def preprocessing():
    preprocessor_Vital = PreprocessingPulseDB(data_path=data_path_Vital, target_path=target_path_Vital, f_low=0.5, f_high=8,
                                              order=4, fs=fs, db="v", replace=True)
    preprocessor_Vital.process()

    preprocessor_MIMIC = PreprocessingPulseDB(data_path=data_path_MIMIC, target_path=target_path_MIMIC, f_low=0.5, f_high=8,
                                              order=4, fs=fs, db="m", replace=True)
    preprocessor_MIMIC.process()

    preprocessor_test = PreprocessingPulseDB(data_path=data_path_test, target_path=target_path_test, f_low=0.5,
                                             f_high=8, order=4, fs=fs, db="v", replace=True)
    # preprocessor_test.process()

    preprocessor_train = PreprocessingPulseDB(data_path=data_path_train, target_path=target_path_train, f_low=0.5,
                                              f_high=8, order=4, fs=fs, db="m", replace=True)
    # preprocessor_train.process()


def training():
    path_main_train = "D:/Download/Vital_proc/"
    train = Train(path_main=path_main_train)
    train.training()


def testing():
    path_main_test = "D:/Download/PulseDB_proc_test/"
    test = Test(path_main=path_main_test)
    test.testing()


if __name__ == '__main__':
    # preprocessing()
    training()
    # testing()
