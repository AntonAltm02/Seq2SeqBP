import os
import torch
from Implementierung.data_generator import DataGenerator
import matplotlib.pyplot as plt
import numpy as np
import scipy
import neurokit2


def plot_output(pred, ref):
    t = np.linspace(0, 8, 1000)

    pred = pred.detach().numpy()
    ref = ref.detach().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))
    ax[0].plot(t, pred[0, :, :])
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Blood pressure (mmHg)")
    ax[0].set_title("Predicted waveform")

    ax[1].plot(t, ref[0, :, :])
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Blood pressure (mmHg)")
    ax[1].set_title("Reference waveform")
    fig.suptitle("Training")

    plt.show()


def calc_sbp_dbp(pred, ref_sbp, ref_dbp):
    pred = pred.squeeze(-1)
    ref_sbp = ref_sbp.squeeze(-1)
    ref_dbp = ref_dbp.squeeze(-1)

    # Convert tensors to NumPy arrays
    pred = pred.detach().numpy()
    ref_sbp = ref_sbp
    ref_dbp = ref_dbp

    pred_sbp = []
    pred_dbp = []

    for i in range(len(pred)):
        extract_max, _ = neurokit2.ppg_process(pred[i, :], sampling_rate=125)
        extract_min, _ = neurokit2.ppg_process(-pred[i, :], sampling_rate=125)

        idx_max = np.where(extract_max.PPG_Peaks == 1)[0]
        idx_min = np.where(extract_min.PPG_Peaks == 1)[0]

        pred_sbp.append(np.mean(pred[i, idx_max]))
        pred_dbp.append(np.mean(pred[i, idx_min]))

    pred_sbp = np.vstack(pred_sbp)
    pred_dbp = np.vstack(pred_dbp)

    # Calculate mean error
    error_sbp = pred_sbp - ref_sbp
    me_sbp = np.mean(error_sbp)
    error_dbp = pred_dbp - ref_dbp
    me_dbp = np.mean(error_dbp)
    print("Mean Error SBP:", me_sbp, "Mean Error DBP:", me_dbp)

    # Calculate mean absolute error
    ae_sbp = np.abs(pred_sbp - ref_sbp)
    mae_sbp = np.mean(ae_sbp)
    ae_dbp = np.abs(pred_dbp - ref_dbp)
    mae_dbp = np.mean(ae_dbp)
    print("Mean Absolute Error SBP:", mae_sbp, "Mean Absolute Error DBP:", mae_dbp)

    # Calculate standard deviation of errors
    std_sbp = np.std(error_sbp)
    std_dbp = np.std(error_dbp)
    print("Standard Deviation SBP:", std_sbp, "Standard Deviation DBP:", std_dbp)

    return pred_sbp, pred_dbp


def corr_plot_sbp(pred_sbp, ref_sbp):
    plt.figure()
    plt.plot(ref_sbp, pred_sbp, marker='o', color='blue', linestyle='None', alpha=0.5)

    # Add labels and title
    plt.xlabel('Reference SBP')
    plt.ylabel('Predicted SBP')
    plt.title('Correlation Plot - SBP')

    # linear relationship
    z = np.polyfit(ref_sbp.squeeze(-1), pred_sbp.squeeze(-1), 1)
    p = np.poly1d(z)
    plt.plot(ref_sbp, p(ref_sbp), color='red')

    corr_coefficient, _ = scipy.stats.pearsonr(ref_sbp.flatten(), pred_sbp.flatten())
    plt.text(0.1, 0.9, f'Correlation Coefficient: {corr_coefficient:.2f}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    plt.show()


def corr_plot_dbp(pred_dbp, ref_dbp):
    plt.figure()
    plt.plot(ref_dbp, pred_dbp, marker='o', color='blue', linestyle='None', alpha=0.5)

    # Add labels and title
    plt.xlabel('Reference DBP')
    plt.ylabel('Predicted DBP')
    plt.title('Correlation Plot - DBP')

    # linear relationship
    z = np.polyfit(ref_dbp.squeeze(-1), pred_dbp.squeeze(-1), 1)
    p = np.poly1d(z)
    plt.plot(ref_dbp, p(ref_dbp), color='red')

    corr_coefficient, _ = scipy.stats.pearsonr(ref_dbp.flatten(), pred_dbp.flatten())
    plt.text(0.1, 0.9, f'Correlation Coefficient: {corr_coefficient:.2f}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    plt.show()


def bland_altman_sbp(predicted, reference, title, xlabel, ylabel):
    mean = np.mean([predicted, reference], axis=0)
    diff = predicted - reference
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, axis=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, color='blue', s=10)
    plt.axhline(mean_diff, color='red', linestyle='--')
    plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


def calc_corr(pred, ref):
    x = pred.squeeze(-1)
    y = ref.squeeze(-1)

    # Convert tensors to NumPy arrays
    x = x.detach().numpy()
    y = y.detach().numpy()

    # Initialize lists to store correlation coefficients
    pearson_corr_list = []

    # Compute Pearson correlation coefficient for each batch
    for i in range(len(pred)):
        pearson_corr, _ = scipy.stats.pearsonr(x[i], y[i])
        pearson_corr_list.append(pearson_corr)

    # Convert list to PyTorch tensor
    pearson_corr_tensor = torch.tensor(pearson_corr_list)

    # Calculate mean and standard deviation
    mean_corr = pearson_corr_tensor.mean().item()
    std_corr = pearson_corr_tensor.std().item()

    print("Mean correlation coefficient:", mean_corr)
    print("Standard deviation:", std_corr)


class Test:
    def __init__(self, path_main):
        self.path_main = path_main
        self.files = os.listdir(path_main + "data/")
        self.batch_size = 32

    def testing(self):
        informer = torch.load(
            "Models/V5_wholeModel_informer_4EncoderLayer_Distilling_CrossAttentionDecoder_3Epochs_AdamW_dmodel512_nheads8_batchsize32.pth")

        test_id = self.files
        generator_test = DataGenerator(path_main=self.path_main, list_id=test_id, batch_size=self.batch_size,
                                       shuffle=True)
        criterion = torch.nn.L1Loss()

        all_pred = []
        all_ref = []
        all_sbp = []
        all_dbp = []

        print("Starting Testing")
        informer.eval()
        with torch.no_grad():  # No need to calculate gradients during inference
            running_loss = 0.0
            batch_num = 0
            for inputs, targets, sbp, dbp in generator_test:
                # if batch_num > 2:
                    # break
                # Forward pass
                outputs = informer(inputs)

                # Calculate loss (if applicable)
                loss_mae = criterion(outputs, targets)
                total_loss = loss_mae

                batch_num += 1
                running_loss += total_loss.item()

                # plot_output(outputs, targets)

                all_pred.append(outputs)
                all_ref.append(targets)
                all_sbp.append(sbp)
                all_dbp.append(dbp)

                print(f'Testing - Batch: {batch_num}, MAE: {total_loss} mmHg')

        avg_loss = running_loss / batch_num
        print(f'Testing, Avg MAE: {avg_loss} mmHg')

        pred = torch.cat(all_pred, dim=0)
        ref = torch.cat(all_ref, dim=0)
        ref_sbp = torch.cat(all_sbp, dim=0).detach().numpy()
        ref_dbp = torch.cat(all_dbp, dim=0).detach().numpy()

        print("")
        # calculation of correlation between the predictions and reference
        calc_corr(pred, ref)

        print("")
        # calculation of sbp and dbp of the predictions
        pred_sbp, pred_dbp = calc_sbp_dbp(pred=pred, ref_sbp=ref_sbp, ref_dbp=ref_dbp)

        # visualization correlation
        corr_plot_sbp(pred_sbp=pred_sbp, ref_sbp=ref_sbp)
        corr_plot_dbp(pred_dbp=pred_dbp, ref_dbp=ref_dbp)

        # bland-altman
        bland_altman_sbp(predicted=pred_sbp, reference=ref_sbp, title="Bland-Altman SBP", xlabel="Average SBP", ylabel="Difference SBP")
        bland_altman_sbp(predicted=pred_dbp, reference=ref_dbp, title="Bland-Altman DBP", xlabel="Average DBP", ylabel="Difference DBP")
