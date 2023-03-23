import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st = 0
rows = 101 - st
runs = 10

train_loss_mean = np.zeros((rows, 1))
valid_loss_mean = np.zeros((rows, 1))
train_loss_item = np.zeros((runs, rows, 1))
valid_loss_item = np.zeros((runs, rows, 1))
train_loss_std = np.zeros((rows, 1))
valid_loss_std = np.zeros((rows, 1))
datasets = ["NASA", "XJTU"]
hvs = ["h", "v"]
for dataset in datasets:
    for hv in hvs:
        if hv == "h" and dataset == "NASA":
            continue
        path = f"./Results/{dataset}/Deep_Wiener/data{hv}_log"
        if dataset == "NASA":
            exp_path = "dw_wiener_alpinstate_alptypegru_alprnn@32@32@32@32_alpmlp@16@16@16_alpbidFalse_alpnhead1_omginstate_omginitb0.01_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_seed"
        elif hv == "v":
            exp_path = "dw_wiener_alpinstate_alptypegru_alprnn@32@32@32@32_alpmlp@16@16@16_alpbidFalse_alpnhead1_omginstate_omginitb1_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_seed"
        else:
            exp_path = "dw_wiener_alpinstate_alptypegru_alprnn@32_alpmlp@16@16@16_alpbidFalse_alpnhead1_omginstate_omginitb0.01_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_lrdr0.9_lr0.002_kfold1_lamd1_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_seed"
        output_path = "./csvs"
        min_dw_step_err = 2e9
        max_wn_step_err = -2e9
        for i in range(runs):
            data_folder = f"{path}/{exp_path}{i}"
            step_err = pd.read_csv(f"{data_folder}/{hv}_cd_diagram.csv", header=0)
            # valid_loss = pd.read_csv(f"{data_folder}/{hv}_ste_cd_diagram.csv", header=0)
            min_dw_step_err = min(min_dw_step_err, step_err.iloc[st:, 0].sum())
            max_wn_step_err = max(max_wn_step_err, step_err.iloc[st:, 1].sum())
            # train_loss_item[i] = train_loss.iloc[st:, 0].values.reshape(rows, 1)
            # valid_loss_item[i] = valid_loss.iloc[st:, 0].values.reshape(rows, 1)
        print(f"min_dw_step_err: {min_dw_step_err}")
        print(f"max_wn_step_err: {max_wn_step_err}")
        print(f"min_dw_step_err / max_wn_step_err: {min_dw_step_err / max_wn_step_err}")
        # train_loss_mean /= runs
        # valid_loss_mean /= runs
        # for i in range(runs):
        #     train_loss_std += (train_loss_item[i] - train_loss_mean)**2
        #     valid_loss_std += (valid_loss_item[i] - valid_loss_mean)**2
        # train_loss_std = np.sqrt(train_loss_std / runs)
        # valid_loss_std = np.sqrt(valid_loss_std / runs)

        # means = [train_loss_mean, valid_loss_mean]
        # stds = [train_loss_std, valid_loss_std]
        #
        # pd.DataFrame({'Train Loss Mean': train_loss_mean.reshape((rows,)), 'Valid Loss Mean': valid_loss_mean.reshape((rows,))}).to_csv(f"{output_path}/{dataset}_learning_curve_{hv}.csv")

        # #### plot ####
        # for i in range(2):
        #     mean = means[i]
        #     std = stds[i]
        #     x = list(range(rows))
        #     y = mean.reshape((rows,))
        #     yerr = std.reshape((rows,))
        #     fig, ax = plt.subplots()
        #     # yerr = [0]
        #     # ax.errorbar(x, y, yerr, linewidth=2, capsize=6)
        #     ax.plot(x,y)
        #     # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
        #     #        ylim=(0, 8), yticks=np.arange(1, 8))
        #
        #     plt.show()
        #
        #
        # for i in range(2):
        #     mean = means[i]
        #     std = stds[i]
        #     x = list(range(rows))
        #     y = mean.reshape((rows,))
        #     yerr = std.reshape((rows,))
        #     fig, ax = plt.subplots()
        #     # yerr = [0]
        #     ax.errorbar(x, y, yerr, linewidth=2, capsize=6)
        #     # ax.plot(x,y)
        #     # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
        #     #        ylim=(0, 8), yticks=np.arange(1, 8))
        #
        #     plt.show()