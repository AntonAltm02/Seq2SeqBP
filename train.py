import os
import numpy as np
import wandb
from sklearn.model_selection import KFold, train_test_split
import torch
from data_generator import DataGenerator
from informer import Informer
import matplotlib.pyplot as plt


def plot_output(pred, ref, run, mode):
    t = np.linspace(0, 8, 1000)
    pred = pred.detach().numpy()
    ref = ref.detach().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(40, 10))

    if mode == "training":
        ax[0].plot(t, pred[0, :, :])
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Blood pressure (mmHg)")
        ax[0].set_title("Predicted waveform")

        ax[1].plot(t, ref[0, :, :])
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Blood pressure (mmHg)")
        ax[1].set_title("Reference waveform")
        fig.suptitle("Training")

        run.log({"Training": wandb.Image(fig)})
        plt.close(fig)

    elif mode == "validation":
        ax[0].plot(t, pred[0, :, :])
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Blood pressure (mmHg)")
        ax[0].set_title("Predicted waveform")

        ax[1].plot(t, ref[0, :, :])
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Blood pressure (mmHg)")
        ax[1].set_title("Reference waveform")
        fig.suptitle("Validation")

        run.log({"Validation": wandb.Image(fig)})
        plt.close(fig)


class Train:
    def __init__(self, path_main):
        self.path_main = path_main
        self.files = os.listdir(path_main + "data/")
        self.n_splits = 3
        self.kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.kfold.get_n_splits(self.files)
        self.batch_size = 32

    def training(self):

        wandb.login(key="7b3a9192b79e86c42a5861948e67a86482e0abd2")

        # start logging for cloud service weights and biases
        run = wandb.init(
            project="projektarbeit2",
            config={
                "learning_rate": 0.001,
                "architecture": "Informer",
                "dataset": "PulseDB",
                "epochs": 1,
            }
        )

        # Informer Neural Network
        informer = Informer(d_model=512, num_head=8, len_seq=1000)
        optimizer = torch.optim.AdamW(informer.parameters(), lr=0.001, weight_decay=0.0001)
        criterion = torch.nn.L1Loss()

        informer.to("cuda")
        print(next(informer.parameters()).device)

        train_id, val_id = train_test_split(self.files, test_size=0.3)

        epochs = 3
        for epoch in range(epochs):
            print("Loading DataGenerator")
            generator_train = DataGenerator(path_main=self.path_main, list_id=train_id, batch_size=self.batch_size,
                                            shuffle=True)

            print(f"Starting training - Epoch: {epoch + 1}")
            informer.train()
            running_loss = 0.0
            batch_num = 0
            for inputs, targets, sbp, dbp in generator_train:
                optimizer.zero_grad()

                outputs = informer(inputs)

                loss_mae = criterion(outputs, targets)
                total_loss = loss_mae
                total_loss.backward()
                optimizer.step()

                batch_num += 1
                running_loss += total_loss.item()

                plot_output(outputs, targets, run, "training")

                print(f'Training - Batch: {batch_num}, MAE: {total_loss} mmHg')
                run.log({"Training-Batch-MAE": total_loss})

            avg_loss = running_loss / batch_num
            print(f'Training - Epoch {epoch + 1}/{epochs}, Avg MAE per Batch: {avg_loss} mmHg')
            run.log({"Training-Epoch-AvgMAE": avg_loss})

            print("Starting validation")
            generator_test = DataGenerator(path_main=self.path_main, list_id=val_id, batch_size=self.batch_size,
                                           shuffle=True)
            informer.eval()
            with torch.no_grad():
                running_loss = 0.0
                batch_num = 0
                for inputs, targets, sbp, dbp in generator_test:
                    outputs = informer(inputs)

                    loss_mae = criterion(outputs, targets)
                    total_loss = loss_mae

                    batch_num += 1
                    running_loss += total_loss.item()

                    plot_output(outputs, targets, run, "validation")

                avg_loss = running_loss / batch_num
                print(f'Validation - Avg MAE per Batch: {avg_loss} mmHg')
                run.log({"Validation-Epoch-AvgMAE": avg_loss})

        wandb.finish()

        model_path = "Models/V5_wholeModel_informer_4EncoderLayer_Distilling_CrossAttentionDecoder_3Epochs_AdamW_dmodel512_nheads8_batchsize32.pth"
        torch.save(informer, model_path)
