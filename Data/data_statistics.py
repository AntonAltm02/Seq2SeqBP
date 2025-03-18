import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

target_path = "D:/PulseDB/Data_proc/"


def plot_prob_distribution(set_id, name):
    stacked_sbp = np.array([])
    stacked_dbp = np.array([])
    for idx in set_id:
        sbp = np.load(target_path + "sbp/" + idx, allow_pickle=True)
        dbp = np.load(target_path + "dbp/" + idx, allow_pickle=True)
        sbp = sbp.reshape(len(sbp), 1)
        dbp = dbp.reshape(len(dbp), 1)

        if stacked_sbp.size == 0:  # Check if stacked_sbp is empty
            stacked_sbp = sbp
        else:
            stacked_sbp = np.vstack((stacked_sbp, sbp))

        if stacked_dbp.size == 0:  # Check if stacked_dbp is empty
            stacked_dbp = dbp
        else:
            stacked_dbp = np.vstack((stacked_dbp, dbp))

    plt.figure(figsize=(12, 6))
    dbp = sns.histplot(stacked_dbp.flatten(), color='red', bins=100, stat="density", label='Diastolic Blood Pressure')
    sbp = sns.histplot(stacked_sbp.flatten(), color='blue', bins=100, stat="density", label='Systolic Blood Pressure')
    plt.xlabel("BP (mmHg)")
    plt.ylabel("Probability")
    plt.title('Blood Pressure Distribution')
    plt.legend(title="Blood Pressure", labels=["DBP", "SBP"])
    plt.tight_layout()
    output_dir = 'Plots/'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists or create it
    plt.savefig(os.path.join(output_dir, f"{name}.pdf"), format='pdf')
    plt.show()
    plt.close()

    print(f"SBP (Mean +- SD mmHg): {np.mean(stacked_sbp.flatten())} +- {np.std(stacked_sbp.flatten())}, DBP (Mean +- SD mmHg): {np.mean(stacked_dbp.flatten())} +- {np.std(stacked_dbp.flatten())}")


if __name__ == "__main__":
    files_id = os.listdir(target_path + "data/")
    train_id, test_id = train_test_split(files_id, test_size=0.3, random_state=42, shuffle=False)
    train_id, val_id = train_test_split(train_id, test_size=0.3, random_state=42, shuffle=False)

    plot_prob_distribution(train_id, name="dist_train")
    plot_prob_distribution(test_id, name="dist_test")
    plot_prob_distribution(val_id, name="dist_val")
