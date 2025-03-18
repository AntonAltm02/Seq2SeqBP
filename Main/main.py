from Data.preprocessing_PulseDB import PreprocessingPulseDB
from Main.test import Test
from Main.train import Train
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.model_selection import train_test_split

# data_path_MIMIC = "D:/PulseDB/PulseDB_MIMIC/"
data_path_Vital = os.path.join("C:/Users/anton/OneDrive - Karolinska Institutet/PulseDB/VitalDB/")
# target_path_MIMIC = "D:/PulseDB/MIMIC_proc/"
target_path_Vital = os.path.join("C:/Users/anton/OneDrive - Karolinska Institutet/PulseDB/VitalDB_proc/")
fs = 125


def preprocessing():
    """
    preprocessor_MIMIC = PreprocessingPulseDB(data_path=data_path_MIMIC, target_path=target_path_MIMIC, f_low=0.5, f_high=8,
                                              order=4, fs=fs, db="m", replace=False)
    preprocessor_MIMIC.process()
    """

    preprocessor_vital = PreprocessingPulseDB(data_path=data_path_Vital, target_path=target_path_Vital,
                                              f_low=0.5, f_high=8, order=4, fs=fs, db="v", replace=False)
    preprocessor_vital.process()


if __name__ == '__main__':
    preprocessing()

    path_main = os.path.join("C:/Users/anton/OneDrive - Karolinska Institutet/PulseDB/VitalDB_proc/")
    files = os.listdir(path_main + "data/")
    train_id, test_id = train_test_split(files, test_size=0.1, random_state=42)

    train = Train(path_main=path_main, files=train_id)
    train.training()

    test = Test(path_main=path_main, files=test_id)
    test.testing()
