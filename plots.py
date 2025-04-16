import pandas as pd
import matplotlib.pyplot as plt

# Alpha values and file naming
alphas = [0.1, 0.5, 0.8]
file_template = "logs/Cifar10_training_log_PF_lrsigma0.01_lrhead0.01_alpha{}.csv"

# Containers for metrics
acc_list = []
nll_list = []
ece_list = []

# Read the metrics from each file
for alpha in alphas:
    filename = file_template.format(alpha)
    df = pd.read_csv(filename)
    
    acc_list.append(df["test_acc"].values[-1])
    nll_list.append(df["test_nll"].values[-1])
    ece_list.append(df["test_ece"].values[-1])

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Accuracy plot
axs[0].plot(alphas, acc_list, marker='o', color='blue')
axs[0].set_title("Accuracy vs Alpha")
axs[0].set_xlabel("Alpha")
axs[0].set_ylabel("Accuracy (%)")
axs[0].grid(True)

# NLL plot
axs[1].plot(alphas, nll_list, marker='s', color='green')
axs[1].set_title("NLL vs Alpha")
axs[1].set_xlabel("Alpha")
axs[1].set_ylabel("Negative Log-Likelihood")
axs[1].grid(True)

# ECE plot
axs[2].plot(alphas, ece_list, marker='^', color='red')
axs[2].set_title("ECE vs Alpha")
axs[2].set_xlabel("Alpha")
axs[2].set_ylabel("Expected Calibration Error")
axs[2].grid(True)

plt.suptitle("ProbFormer Test Metrics Across Different Alpha Values", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
plt.savefig("Ablation_C10.png")
plt.show()
