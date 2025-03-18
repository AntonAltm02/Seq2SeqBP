import numpy as np
import random
from keras.utils import Sequence
import torch


class DataGenerator(Sequence):
    def __init__(self, path_main, list_id, batch_size, n_sample=1000, n_classes=1000, shuffle=True):
        ##
        # @brief This constructor initializes the DataGenerator object.
        # @param path_main      The main path which includes the preprocessed data.
        # @param list_id        The list of the subject IDs.                 
        # @param batch_size     The size of each data batch.
        # @param n_sample       The number of samples in each data instance. Default is 624.
        # @param n_classes      The number of output classes. Default is 2.
        # @param shuffle        Whether to shuffle the data after each epoch. Default is True.

        self.path_main = path_main
        self.list_id = list_id
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n_sample = n_sample

        ## @brief Index of the actual ID from the parameter list_id. 
        ##
        self._id_idx = 0
        ## @brief Total number of batches. 
        ##        
        self._nr_batches = 0
        ## @brief Last index of periods of actual subject.
        ##
        self.inPatientIdx = 0
        ## @brief Input data of actual subject.
        ##
        self._dev0 = np.load(self.path_main + "data/" + self.list_id[0], allow_pickle=True)
        ## @brief Target data of actual subject.
        ##
        self._target = np.load(self.path_main + "ground_truth/" + self.list_id[0], allow_pickle=True)
        ## @brief SBP data of actual subject.
        ##
        self._sbp = np.load(self.path_main + "sbp/" + self.list_id[0], allow_pickle=True)
        ## @brief DBP data of actual subject.
        ##
        self._dbp = np.load(self.path_main + "dbp/" + self.list_id[0], allow_pickle=True)

    def __count_batches(self):
        ##
        # @brief    This method count the total number of batches.
        # @return   Total number of batches.
        ##
        if self._nr_batches == 0:
            # print("Counting Subjects")
            rest = 0
            for nr, sub in enumerate(self.list_id):
                if nr != len(self.list_id) - 1:
                    n_seg = len(np.load(self.path_main + "data/" + sub))
                    # print(n_seg)
                    temp_b1, temp_r = divmod(n_seg, self.batch_size)
                    temp_b2, rest = divmod(temp_r + rest, self.batch_size)
                    self._nr_batches += temp_b1 + temp_b2
                else:
                    n_seg = len(np.load(self.path_main + "data/" + sub))
                    # print(n_seg)
                    temp_b1, temp_r = divmod(n_seg, self.batch_size)
                    temp_b2, rest = divmod(temp_r + rest, self.batch_size)
                    if rest > 0:
                        self._nr_batches += temp_b1 + temp_b2 + 1
                    else:
                        self._nr_batches += temp_b1 + temp_b2

        return self._nr_batches

    def __len__(self):
        ##
        # @brief This method count the total number of batches.
        # @return Total number of batches.
        ##
        return self.__count_batches()

    def __getitem__(self, idx):
        ##
        # @brief This method returns a batch of data.
        # @return A batch of data
        ##
        data, target, sbp, dbp = self.__data_generation()
        return data, target, sbp, dbp

    def on_epoch_end(self):
        ## 
        # @brief This method updates the indexes after each epoch.
        ##
        if self.shuffle:
            random.shuffle(self.list_id)
        self.inPatientIdx = 0
        self._id_idx = 0
        self.__load_data()

    def __data_generation(self):
        ##
        # @brief This method generate one batch.
        # @return A batch of data
        ##
        data = np.zeros((self.batch_size, self.n_sample))
        target = np.zeros((self.batch_size, self.n_classes))
        sbp = np.zeros((self.batch_size, 1))
        dbp = np.zeros((self.batch_size, 1))

        for i in range(self.batch_size):

            if self.inPatientIdx == len(self._dev0):
                if self._id_idx == len(self.list_id) - 1:
                    break
                else:
                    self._id_idx += 1
                    self.inPatientIdx = 0
                    self.__load_data()

            data[i] = self._dev0[self.inPatientIdx]
            target[i] = self._target[self.inPatientIdx]
            sbp[i] = self._sbp[self.inPatientIdx]
            dbp[i] = self._dbp[self.inPatientIdx]
            self.inPatientIdx += 1

        non_zero_rows = ~np.all(data == 0, axis=1)

        data = data[non_zero_rows]
        target = target[non_zero_rows]
        sbp = sbp[non_zero_rows]
        dbp = dbp[non_zero_rows]

        data = torch.tensor(data).unsqueeze(-1).to(dtype=torch.float32)
        target = torch.tensor(target).unsqueeze(-1).to(dtype=torch.float32)
        sbp = torch.tensor(sbp).to(dtype=torch.float32)
        dbp = torch.tensor(dbp).to(dtype=torch.float32)

        return data, target, sbp, dbp

    def __load_data(self):
        ##
        # @brief This method loads data of the next subject.
        ##   
        self._dev0 = np.load(self.path_main + "data/" + self.list_id[self._id_idx], allow_pickle=True)
        self._target = np.load(self.path_main + "ground_truth/" + self.list_id[self._id_idx], allow_pickle=True)
        self._sbp = np.load(self.path_main + "sbp/" + self.list_id[self._id_idx], allow_pickle=True)
        self._dbp = np.load(self.path_main + "dbp/" + self.list_id[self._id_idx], allow_pickle=True)
