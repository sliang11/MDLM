import os
import sys
import pandas as pd
import numpy as np
from utils.tools import mkdir

import matplotlib.pyplot as plt

st = 0
rows = 101 - st
runs = 10

# train_loss_mean = np.zeros((rows, 1))
# valid_loss_mean = np.zeros((rows, 1))
# train_loss_item = np.zeros((runs, rows, 1))
# valid_loss_item = np.zeros((runs, rows, 1))
# train_loss_std = np.zeros((rows, 1))
# valid_loss_std = np.zeros((rows, 1))
deepwiener_err_mean = 0
wiener_err_mean = 0
deepwiener_err_item = np.zeros(runs)
wiener_err_item = np.zeros(runs)
train_time_item = np.zeros(runs)
test_time_item = np.zeros(runs)
deepwiener_err_std = 0
wiener_err_std = 0

datasets = ["NASA", "XJTU"]
hvs = ["h", "v"]
for dataset in datasets:
    for hv in hvs:
        if hv == "h" and dataset == "NASA":
            continue
        path = f"./Results_20230201_runs/{dataset}_removeFalse/Deep_Wiener/data{hv}_log_best"
        if dataset == "NASA":
            exp_path = "dw_wiener_alpinstate_alptypegru_alprnn@16@16_alpmlp@16_alpbidFalse_alpdrop0_alpdiff2_alpnhead1_alpwndFalse_omginstate_omginitb0.3_errmlp@16_errOnormTrue_errdrop0_initsgm0.01_concatn3_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_seed"
        elif hv == "v":
            exp_path = "dw_wiener_alpinstate_alptypegru_alprnn@16@16_alpmlp_alpbidTrue_alpdrop0_alpdiff2_alpnhead3_alpwndFalse_omginstate_omginitb0.1_errmlp@32_errOnormTrue_errdrop0_initsgm0.1_concatn3_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_seed"
        else:
            exp_path = "dw_wiener_alpinstate_alptypegru_alprnn@16@16_alpmlp@16_alpbidFalse_alpdrop0_alpdiff2_alpnhead1_alpwndFalse_omginstate_omginitb0.3_errmlp@16_errOnormTrue_errdrop0_initsgm0.01_concatn3_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_seed"
        output_path = f"./csvs/{dataset}_data{hv}/{exp_path}"
        mkdir(output_path)
        for i in range(runs):
            data_folder = f"{path}/{exp_path}{i}"
            errs = pd.read_csv(f"{data_folder}/err_sum.csv", header=0, index_col=0)
            times = pd.read_csv(f"{data_folder}/time.csv", header=0, index_col=0)

            # valid_loss = pd.read_csv(f"{data_folder}/err_sum.csv", header=0)
            # deepwiener_err_mean += errs.values[1][0]
            # wiener_err_mean += errs.values[1][1]
            deepwiener_err_item[i] = errs.values[1][0]
            wiener_err_item[i] = errs.values[1][1]
            train_time_item[i] = times.values[0][0]
            test_time_item[i] = times.values[1][0]
        deepwiener_err_mean = np.mean(deepwiener_err_item)
        wiener_err_mean = np.mean(wiener_err_item)
        # for i in range(runs):
        deepwiener_err_std = np.std(deepwiener_err_item)
        wiener_err_std = np.std(wiener_err_item)
        # deepwiener_err_std = np.sqrt(deepwiener_err_std / runs)
        # wiener_err_std = np.sqrt(wiener_err_std / runs)
        deepwiener_err_best = np.min(deepwiener_err_item)
        deepwiener_err_worst = np.max(deepwiener_err_item)
        wiener_err_best = np.min(wiener_err_item)
        wiener_err_worst = np.max(wiener_err_item)
        train_time_mean = np.mean(train_time_item)
        test_time_mean = np.mean(test_time_item)
        train_time_std = np.std(train_time_item)
        test_time_std = np.std(test_time_item)


        deepwiener = [deepwiener_err_mean, deepwiener_err_std, deepwiener_err_best, deepwiener_err_worst]
        wiener = [wiener_err_mean, wiener_err_std, wiener_err_best, wiener_err_worst]
        stat = [(wiener_err_mean-deepwiener_err_mean)/wiener_err_mean,
                (wiener_err_best-deepwiener_err_worst)/wiener_err_best,
                (wiener_err_best-deepwiener_err_best)/wiener_err_best,
                (wiener_err_worst-deepwiener_err_worst)/wiener_err_worst]
        train_time = [train_time_mean, train_time_std]
        test_time = [test_time_mean, test_time_std]
        pd.DataFrame({'DeepWiener': deepwiener, 'Wiener': wiener}, index=['Mean', 'Std', 'Best', 'Worst']).to_csv(
            f"{output_path}/data.csv")
        pd.DataFrame({'Stat': stat}, index=['Mean Gain', 'Worst Best Gain', 'Best Gain', 'Worst Gain']).to_csv(
            f"{output_path}/stat.csv")
        pd.DataFrame({'Train time': train_time, 'Test time': test_time}, index=['Mean', 'Std']).to_csv(
            f"{output_path}/time.csv")


        # for i in range(runs):
        #     data_folder = f"{path}/{exp_path}{i}"
        #     train_loss = pd.read_csv(f"{data_folder}/{hv}_trainloss.csv", header=0)
        #     valid_loss = pd.read_csv(f"{data_folder}/{hv}_validloss.csv", header=0)
        #     train_loss_mean += train_loss.iloc[st:, 0].values.reshape(rows, 1)
        #     valid_loss_mean += valid_loss.iloc[st:, 0].values.reshape(rows, 1)
        #     train_loss_item[i] = train_loss.iloc[st:, 0].values.reshape(rows, 1)
        #     valid_loss_item[i] = valid_loss.iloc[st:, 0].values.reshape(rows, 1)
        # train_loss_mean /= runs
        # valid_loss_mean /= runs
        # for i in range(runs):
        #     train_loss_std += (train_loss_item[i] - train_loss_mean)**2
        #     valid_loss_std += (valid_loss_item[i] - valid_loss_mean)**2
        # train_loss_std = np.sqrt(train_loss_std / runs)
        # valid_loss_std = np.sqrt(valid_loss_std / runs)
        #
        # means = [train_loss_mean, valid_loss_mean]
        # stds = [train_loss_std, valid_loss_std]
        #
        # pd.DataFrame({'Train Loss Mean': train_loss_mean.reshape((rows,)), 'Valid Loss Mean': valid_loss_mean.reshape((rows,))}).to_csv(f"{output_path}/{dataset}_learning_curve_{hv}.csv")

        #### plot ####
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
