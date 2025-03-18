import os
import torch
from Data.data_generator import DataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import scipy
import neurokit2
import seaborn as sns


def plot_output(pred, ref):
    """
    Plots the predicted and reference ABP signals during testing
    :param pred: predicted ABP signal
    :param ref: reference ABP signal
    :return:
    """
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


def calc_sbp_dbp(pred, ref):
    """
    Calculates the SBP and DBP values for the given ABP signals of predicted and reference waveforms
    Additionally calculates the MAE, ME and SDE between the calculated reference and predicted SBP and DBP values
    :param pred: predicted ABP signal
    :param ref: reference ABP signal
    :return:
    """
    pred = pred.squeeze(-1)
    ref = ref.squeeze(-1)

    pred = pred.detach().numpy()
    ref = ref.detach().numpy()

    pred_sbp = []
    pred_dbp = []
    ref_sbp = []
    ref_dbp = []

    def get_peaks(signal):
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

    for i in range(len(pred)):
        sbp, dbp = get_peaks(pred[i, :])
        pred_sbp.append(sbp), pred_dbp.append(dbp)
        sbp, dbp = get_peaks(ref[i, :])
        ref_sbp.append(sbp), ref_dbp.append(dbp)

    pred_sbp = np.vstack(pred_sbp)
    pred_dbp = np.vstack(pred_dbp)
    ref_sbp = np.vstack(ref_sbp)
    ref_dbp = np.vstack(ref_dbp)

    # Calculate mean error
    error_sbp = pred_sbp - ref_sbp
    me_sbp = np.mean(error_sbp)
    error_dbp = pred_dbp - ref_dbp
    me_dbp = np.mean(error_dbp)
    print(f"Mean Error SBP: {me_sbp:.2f}, Mean Error DBP: {me_dbp:.2f}")

    # Calculate mean absolute error
    ae_sbp = np.abs(pred_sbp - ref_sbp)
    mae_sbp = np.mean(ae_sbp)
    ae_dbp = np.abs(pred_dbp - ref_dbp)
    mae_dbp = np.mean(ae_dbp)
    print(f"Mean Absolute Error SBP: {mae_sbp:.2f}, Mean Absolute Error DBP: {mae_dbp:.2f}")

    # Calculate standard deviation of errors
    std_sbp = np.std(error_sbp)
    std_dbp = np.std(error_dbp)
    print(f"Standard Deviation SBP: {std_sbp:.2f}, Standard Deviation DBP: {std_dbp:.2f}")

    return pred_sbp, pred_dbp, ref_sbp, ref_dbp


def corr_plot(pred, ref, bp_type):
    """
    Plots the correlation between the predicted and reference BP values
    :param pred: predicted BP values
    :param ref: reference BP values
    :param bp_type: type of BP values (SBP/DBP)
    :return:
    """
    plt.figure()
    plt.plot(ref, pred, color="blue", marker='o', linestyle='None', alpha=0.5)

    # Add labels and title
    plt.xlabel(f'Reference {bp_type}')
    plt.ylabel(f'Predicted {bp_type}')
    plt.title(f'Correlation Plot - {bp_type}')

    # linear relationship
    z = np.polyfit(ref.squeeze(-1), pred.squeeze(-1), 1)
    p = np.poly1d(z)
    plt.plot(ref, p(ref), color='red')

    corr_coefficient, _ = scipy.stats.pearsonr(ref.flatten(), pred.flatten())
    plt.text(plt.xlim()[0] + 0.05 * (plt.xlim()[1] - plt.xlim()[0]),
             plt.ylim()[1] - 0.05 * (plt.ylim()[1] - plt.ylim()[0]),
             f'R: {corr_coefficient:.2f}', fontsize=12, color="black", ha='left', va='top')

    path = os.path.join("..", "Plots", f"{bp_type}_corr_plot.pdf")
    plt.savefig(path, format='pdf')
    plt.show()


def density_plot(pred, ref, bp_type):
    """
    Plots the distribution of the BP type
    :param pred: predicted BP values
    :param ref: reference BP values
    :param bp_type: type of BP values (SBP/DBP)
    :return:
    """
    plt.figure(figsize=(8, 6))

    sns.histplot(pred.flatten(), color="blue", bins=50, stat="density", label=f'Predicted {bp_type}')
    sns.histplot(ref.flatten(), color="red", bins=50, stat="density", label=f'Reference {bp_type}')

    plt.xlabel("BP (mmHg)")
    plt.ylabel("Probability")
    plt.title('Blood Pressure Distribution')

    plt.legend(title="Blood Pressure", labels=[f"Predicted {bp_type}", f"Reference {bp_type}"])
    plt.tight_layout()
    path = os.path.join("..", "Plots", f"{bp_type}_density.pdf")
    plt.savefig(path, format='pdf')
    plt.show()
    plt.close()


def bland_altman(predicted, reference, bp_type):
    """
    Plot the Bland-Altman plots
    :param predicted: predicted BP value
    :param reference: reference BP value
    :param bp_type: type of BP values that are plotting (SBP or DBP)
    :return:
    """
    mean = np.mean([predicted, reference], axis=0)
    diff = predicted - reference
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, axis=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, color="blue", s=10)
    plt.axhline(mean_diff, color='red', linestyle='--')
    plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')

    plt.title(f"Bland-Altman {bp_type}")
    plt.xlabel(f"Average {bp_type}")
    plt.ylabel(f"Difference {bp_type}")

    path = os.path.join("..", "Plots", f"{bp_type}_bland_altman.pdf")
    plt.savefig(path, format='pdf')
    plt.show()


def calc_corr(pred, ref):
    """
    Calculates the correlation between the predicted and reference ABP signals
    :param pred: predicted ABP signal
    :param ref: reference ABP signal
    :return:
    """
    # squeeze the last dimension/decrease the shape
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

    print(f"mean correlation coefficient: {mean_corr:.2f}, standard deviation: {std_corr:.2f}")


class Test:
    def __init__(self, path_main, files):
        self.path_main = path_main
        self.files = os.listdir(path_main + "data/")
        _, self.test_id = train_test_split(self.files, test_size=0.3, random_state=42)
        self.batch_size = 32

    def testing(self):
        print("Loading Informer...")
        model_path = os.path.join("..", "Models", "V6_finalModel.pth")
        informer = torch.load(model_path)

        print("Lading Data...")
        generator_test = DataGenerator(path_main=self.path_main, list_id=self.test_id, batch_size=self.batch_size,
                                       shuffle=True)
        criterion = torch.nn.L1Loss()

        all_pred = []
        all_ref = []

        print("Starting Testing")
        informer.eval()
        with torch.no_grad():  # No need to calculate gradients during inference
            running_loss = 0.0
            batch_num = 0
            for inputs, targets, sbp, dbp in generator_test:
                # output of the trained Informer
                outputs = informer(inputs)

                # Calculate loss (if applicable)
                loss_mae = criterion(outputs, targets)
                total_loss = loss_mae

                batch_num += 1
                running_loss += total_loss.item()

                all_pred.append(outputs)
                all_ref.append(targets)

                print(f'Testing - Batch: {batch_num}, MAE: {total_loss} mmHg')

        avg_loss = running_loss / batch_num
        print(f'Testing, Avg MAE: {avg_loss} mmHg')

        pred = torch.cat(all_pred, dim=0)
        ref = torch.cat(all_ref, dim=0)

        print("")
        # calculation of correlation between the predictions and reference
        calc_corr(pred, ref)

        print("")
        # calculation of sbp and dbp of the predictions
        pred_sbp, pred_dbp, ref_sbp, ref_dbp = calc_sbp_dbp(pred=pred, ref=ref)

        # visualization correlation
        corr_plot(pred=pred_sbp, ref=ref_sbp, bp_type="SBP")
        corr_plot(pred=pred_dbp, ref=ref_dbp, bp_type="DBP")

        # bland-altman
        bland_altman(predicted=pred_sbp, reference=ref_sbp, bp_type="SBP")
        bland_altman(predicted=pred_dbp, reference=ref_dbp, bp_type="DBP")

        # density plots
        density_plot(pred=pred_sbp, ref=ref_sbp, bp_type="SBP")
        density_plot(pred=pred_dbp, ref=ref_dbp, bp_type="DBP")
