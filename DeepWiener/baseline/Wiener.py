import pickle
import numpy as np
import math
import random
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import copy
import os
import torch.nn as nn
import torch
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import time


# b=1
def load_data(fn):
    """载入数据"""
    with open(fn + '.pickle', "rb") as f:
        x = pickle.load(f)
    return x


def save_data(data, fn):
    with open(fn + '.pickle', "wb") as f:
        pickle.dump(data, f)


def mkdir(path):
    # 判断路径是否存在
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


def omega_function(t, s, b):
    """第一种w公式形式"""
    return t * (b ** (t - 1))


def omega_function_1(t, s, b):
    """第二种w公式形式"""
    return b * np.exp(b * t)


def omega_function_2(t, s, b):
    """第三种w公式形式"""
    return s ** b


def l_b_s(b, train_o, mod):
    """
    似然函数
    :param b:需要搜索求解的变量
    :param train: 训练数据
    :param mod: w的形式
    :return: 似然函数值
    """
    sigma_B_hat_up = 0
    sigma_B_hat_dowm = 0
    sigma_N_all = 0
    an_list = []
    for i in range(len(train_o)):
        train = np.array(train_o[i])
        Sn = np.diff(train)
        t = np.array(range(1, len(Sn) + 2))
        delt_t = np.diff(t)
        SIGMA_N = np.diag(delt_t)
        if mod == 0:
            omega_n = np.apply_along_axis(omega_function, 0, t[:-1], Sn, b)
        if mod == 1:
            # omega_n = np.apply_along_axis(omega_function_1, 0, t[:-1], Sn, b)
            # 这里有点问题,虽然好像能跑通
            omega_n = omega_function_1(t[:-1], Sn, b)
        if mod == 2:
            # omega_n 与状态相关
            omega_n = np.apply_along_axis(omega_function_2, 0, t[:-1], train[:-1], b)
        Kn = len(Sn)
        an_hat = np.dot(np.dot(omega_n.T, np.linalg.inv(SIGMA_N)), Sn) / np.dot(
            np.dot(omega_n.T, np.linalg.inv(SIGMA_N)), omega_n)
        an_list.append(an_hat)
        sigma_B_hat_up += np.dot(np.dot((Sn - an_hat * omega_n).T, np.linalg.inv(SIGMA_N)), (Sn - an_hat * omega_n))
        sigma_B_hat_dowm += Kn
        sigma_N_all += np.linalg.det(SIGMA_N)
    # sigma_B_hat = np.dot(np.dot((Sn - an_hat * omega_n).T, np.linalg.inv(SIGMA_N)), (Sn - an_hat * omega_n)) / (Kn)
    sigma_B_hat = sigma_B_hat_up / sigma_B_hat_dowm
    l_b_S = - (math.log(2 * math.pi) + math.log(sigma_B_hat) + 1) / 2 * sigma_B_hat_dowm - 1 / 2 * math.log(
        sigma_N_all)
    return l_b_S, an_list, sigma_B_hat


def prediction(s_k, a_n, b, sigma_B, delta_t, k):
    """s_k 表示s_{k-1}"""
    gauss = random.gauss(0, delta_t)
    return s_k + a_n * (k * (b ** (k - 1))) + sigma_B * gauss, (k * (b ** (k - 1))), sigma_B * gauss


def prediction_1(s_k, a_n, b, sigma_B, delta_t, k):
    """s_k 表示s_{k-1}"""
    gauss = random.gauss(0, delta_t)
    return s_k + a_n * (b * math.exp(b * k)) + sigma_B * gauss, (b * math.exp(b * k)), sigma_B * gauss


def prediction_2(s_k, a_n, b, sigma_B, delta_t, k):
    """s_k 表示s_{k-1}"""
    gauss = random.gauss(0, delta_t)
    return s_k + a_n * (s_k ** b) + sigma_B * gauss, (s_k ** b), sigma_B * gauss


# def l_b_S(sigma_B,Kn,SIGMA_N):
#     re =  - (math.log(2*math.pi)+math.log(sigma_B)+1)/2 + (Kn-1) - 1/2 *math.log(np.linalg.det(SIGMA_N))
#     return re
# def an_hat_function(w,sigma,delta_s):
#     return (w.T*(sigma^(-1))*delta_s)/(w.T*(sigma^(-1))*w)

# def sigma2B_hat_function(w,sigma,delta_s,Kn):
#     return sigma

# def Sn_function(w,sigma,delta_s,Kn):
#     pass


# train = np.array(v_state_train[0])
# Sn = np.diff(train)
# t = np.array(range(1,len(Sn)+2))
# delt_t = np.diff(t)
# SIGMA_N = np.diag(delt_t)
# omega_n = np.apply_along_axis(omega_function,0,t[:-1],Sn,b)
# Kn = len(Sn)
# an_hat = np.dot(np.dot(omega_n.T,SIGMA_N ),Sn)/np.dot(np.dot(omega_n.T,SIGMA_N ),omega_n)
# sigma_B_hat = np.dot(np.dot((Sn-an_hat*omega_n).T,SIGMA_N ),(Sn-an_hat*omega_n))/(Kn-1)
# l_b_S =  - (math.log(2*math.pi)+math.log(sigma_B_hat)+1)/2 + (Kn-1) - 1/2 *math.log(np.linalg.det(SIGMA_N))
# l  = [ lambda b: l_b_s(b,v_state_train) for b in range(10)]
# res = [m for m in l]


# =====================offline参数估计====================
def parameter_estimation(train, mod, b_range, b_granularity):
    """
    :param train:训练数据
    :param mod: w函数的形式
    :param b_range: 参数b的搜索个数
    :param b_granularity: 参数b的搜索粒度
    for example:如果b_range = range(1,10),b_granularity= 10
    b的真实取值范围为[1.0,1.1,1.2,......,1.9]
    :return: 预测函数
    """
    # b_range = range(9000,9801) #mod=0 参数范围
    # b_range = range(8000,8501) #mod=2 参数范围
    l_b_S_list = []
    an_hat_list = []
    sigma_B_hat_list = []
    #     best_lbs = 9999
    best_lbs = 0
    best_b = 0
    best_an = [0] * len(train)
    best_sigma_B = 0
    idx = 0
    for b in b_range:
        # b_pie = b/1
        # b_pie = b/10000 #mod=0 参数范围
        # b_pie = b / 100000  # mod=2 参数范围

        b_pie = b / b_granularity
        l_b_S, an_hat, sigma_B_hat = l_b_s(b_pie, train, mod)
        #         if idx == 0 or (idx % 100 == 0):
        #         print(f'lbS: {l_b_S}, an_hat: {an_hat}, sigma_B_hat: {sigma_B_hat}')
        l_b_S_list.append(l_b_S)
        an_hat_list.append(an_hat)
        sigma_B_hat_list.append(sigma_B_hat)
        if l_b_S > best_lbs:
            best_lbs = l_b_S
            best_b = b_pie
            best_an = an_hat
            best_sigma_B = sigma_B_hat
        idx += 1
    return best_b, best_an, best_sigma_B, best_lbs


def main(v_state_train, v_state_test, b_V, an_V, sigma_B_V, mod=0, real_value=1):
    """
    :param mod: w函数的形式
    :param b_range: 参数b的搜索个数
    :param b_granularity: 参数b的搜索粒度
    for example:如果b_range = range(1,10),b_granularity= 10
    b的真实取值范围为[1.0,1.1,1.2,......,1.9]
    :param real_value:取值范围为0/1,0表多步预测,1表单步预测
    :return V_TR,H_TR,V_TE,H_TE: 根据公式预测的V的训练集，H的训练集，V的测试集，H的测试集
    """
    #     data_folder = "NASA_Bearing_Datasets"
    #     split_folder = 'Train_Test'
    #     folder = os.path.join(data_folder, split_folder)
    #     v_state_train = load_data(folder + '/v_state_train')
    #     v_state_test = load_data(folder + '/v_state_test')
    #     v_vibr_train = load_data(folder + '/v_vibr_train')
    #     v_vibr_test = load_data(folder + '/v_vibr_test')
    #     h_state_train = load_data(folder + 'h_state_train')
    #     h_state_test = load_data(folder + 'h_state_test')
    #     h_vibr_train = load_data(folder + 'h_vibr_train')
    #     h_vibr_test = load_data(folder + 'h_vibr_test')

    #     b_V, an_V, sigma_B_V, _ = parameter_estimation(v_state_train, mod, b_range, b_granularity)
    #     b_H, an_H, sigma_B_H, _ = parameter_estimation(h_state_train, mod, b_range, b_granularity)

    V_TR = []
    H_TR = []
    V_TE = []
    H_TE = []
    V_omegas = []
    V_errors = []
    H_omegas = []
    H_errors = []

    if (real_value == 0):
        # =================测试训练集=====================================================
        # ===============训练集测试方法为拿第一个点作为已知S_O，根据公式计算后面的值=======================
        for i in range(len(v_state_train)):
            train_V = v_state_train[i]
            train_H = h_state_train[i]
            train_predict_V = np.zeros(len(train_V)).tolist()
            train_predict_H = np.zeros(len(train_H)).tolist()
            omega_V = np.zeros(len(train_V)).tolist()
            error_V = np.zeros(len(train_V)).tolist()
            omega_H = np.zeros(len(train_H)).tolist()
            error_H = np.zeros(len(train_H)).tolist()
            train_predict_V[0] = train_V[0]
            train_predict_H[0] = train_H[0]

            print("predict No. " + str(i) + " train dataset.")
            try:
                for j in range(1, len(train_V)):

                    if (mod == 0):
                        train_predict_V[j], omega_V[j], error_V[j] = prediction(train_predict_V[j - 1], an_V[i], b_V,
                                                                                sigma_B_V, 1, j)
                        p = 1
                        # train_predict_H[j] = prediction(train_predict_H[j - 1], an_H, b_H, sigma_B_H, 1, j)
                    if (mod == 1):
                        train_predict_V[j], omega_V[j], error_V[j] = prediction_1(train_predict_V[j - 1], an_V[i], b_V,
                                                                                  sigma_B_V, 1, j)
                        # train_predict_H[j] = prediction_1(train_predict_H[j - 1], an_H, b_H, sigma_B_H, 1, j )
                    if (mod == 2):
                        train_predict_V[j], omega_V[j], error_V[j] = prediction_2(train_predict_V[j - 1], an_V[i], b_V,
                                                                                  sigma_B_V, 1, j)

                for j in range(1, len(train_H)):

                    if (mod == 0):
                        # train_predict_V[j] = prediction(train_predict_V[j - 1], an_V, b_V, sigma_B_V, 1, j)
                        train_predict_H[j], omega_H[j], error_H[j] = prediction(train_predict_H[j - 1], an_H[i], b_H,
                                                                                sigma_B_H, 1, j)
                    if (mod == 1):
                        # train_predict_V[j] = prediction_1(train_predict_V[j - 1], an_V, b_V, sigma_B_H, 1, j)
                        train_predict_H[j], omega_H[j], error_H[j] = prediction_1(train_predict_H[j - 1], an_H[i], b_H,
                                                                                  sigma_B_H, 1, j)
                    if (mod == 2):
                        # train_predict_V[j] = prediction_2(train_predict_V[j - 1], an_H, b_H, sigma_B_H, 1, j)
                        train_predict_H[j], omega_H[j], error_H[j] = prediction_2(train_predict_H[j - 1], an_H[i], b_H,
                                                                                  sigma_B_H, 1, j)

                V_TR.append(train_predict_V)
                H_TR.append(train_predict_H)

            except Exception as e:
                print("No. " + str(i) + " train dataset get error when prediction.")
            plt.plot(range(len(train_V)), train_predict_V)
            plt.plot(range(len(train_V)), train_V)
            plt.savefig('.\\all-step\\train_prediction\\mod_' + str(mod) + '_No_' + str(i) + '_V_train_predict.jpg')
            plt.show()
            plt.plot(range(len(train_H)), train_predict_H)
            plt.plot(range(len(train_H)), train_H)
            plt.savefig('.\\all-step\\train_prediction\\mod_' + str(mod) + '_No_' + str(i) + '_H_train_predict.jpg')
            plt.show()

        # =================测试测试集=====================================================
        # ================测试集测试方法为最后一个训练集的点作为已知，根据公式计算测试集的值=========
        for i in range(len(v_state_train)):
            train_V = np.array(v_state_train[i])
            train_H = np.array(h_state_train[i])
            test_V = np.array(v_state_test[i])
            test_H = np.array(h_state_test[i])
            data_V = np.concatenate([np.array(v_state_train[i]), np.array(v_state_test[i])])
            data_H = np.concatenate([np.array(h_state_train[i]), np.array(h_state_test[i])])
            test_V_pie = np.array(v_state_test[i])
            test_H_pie = np.array(h_state_test[i])
            test_predict_V = copy.deepcopy(data_V[len(train_V) - 1:])
            test_predict_H = copy.deepcopy(data_H[len(train_H) - 1:])
            test_predict_V_pie = copy.deepcopy(data_V[len(train_V) - 1:])
            test_predict_H_pie = copy.deepcopy(data_H[len(train_H) - 1:])
            # test_predict_V = np.concatenate([np.array(v_state_train[i][-1]),np.array(v_state_test[i])])
            # test_predict_H = np.concatenate([np.array(h_state_train[i][-1]),np.array(h_state_test[i])])

            # test_predict_V[0] = test_V_pie[0]
            # test_predict_H[0] = test_H_pie[0]
            # b_V,an_V,sigma_B_V,_ = parameter_estimation(train_V,mod,b_range,b_granularity)
            # b_H, an_H, sigma_B_H, _ = parameter_estimation(train_H,mod,b_range,b_granularity)
            print("predict No. " + str(i) + " test dataset")
            try:
                for j in range(1, len(test_V) + 1):

                    if (mod == 0):
                        test_predict_V[j], _, _ = prediction(test_predict_V[j - 1], an_V[i], b_V, sigma_B_V, 1,
                                                             j + len(train_V))
                        # test_predict_H[j] = prediction(test_predict_H[j - 1], an_H, b_H, sigma_B_H, 1, j)
                    if (mod == 1):
                        test_predict_V[j], _, _ = prediction_1(test_predict_V[j - 1], an_V[i], b_V, sigma_B_V, 1,
                                                               j + len(train_V))
                        # test_predict_H[j] = prediction_1(test_predict_H[j - 1], an_H, b_H, sigma_B_H, 1, j )
                    if (mod == 2):
                        test_predict_V[j], _, _ = prediction_2(test_predict_V[j - 1], an_V[i], b_V, sigma_B_V, 1,
                                                               j + len(train_V))
                        # test_predict_H[j] = prediction_2(test_predict_H[j - 1], an_H, b_H, sigma_B_H, 1, j )
                for j in range(1, len(test_H) + 1):

                    if (mod == 0):
                        # test_predict_V[j] = prediction(test_predict_V[j - 1], an_V, b_V, sigma_B_V, 1, j)
                        test_predict_H[j], _, _ = prediction(test_predict_H[j - 1], an_H[i], b_H, sigma_B_H, 1,
                                                             j + len(train_H))
                    if (mod == 1):
                        # test_predict_V[j] = prediction_1(test_predict_V[j - 1], an_V, b_V, sigma_B_V, 1, j)
                        test_predict_H[j], _, _ = prediction_1(test_predict_H[j - 1], an_H[i], b_H, sigma_B_H, 1,
                                                               j + len(train_H))
                    if (mod == 2):
                        # test_predict_V[j] = prediction_2(test_predict_V[j - 1], an_V, b_V, sigma_B_V, 1, j)
                        test_predict_H[j], _, _ = prediction_2(test_predict_H[j - 1], an_H[i], b_H, sigma_B_H, 1,
                                                               j + len(train_H))

                V_TE.append(test_predict_V)
                H_TE.append(test_predict_H)

            except Exception as e:
                print("No. " + str(i) + " train dataset get error when prediction.")
            plt.plot(range(len(test_V)), test_predict_V[1:])
            plt.plot(range(len(test_V)), test_V)
            plt.savefig('.\\all-step\\test_prediction\\mod_' + str(mod) + '_No_' + str(i) + '_V_test_predict.jpg')
            plt.show()
            plt.plot(range(len(test_H)), test_predict_H[1:])
            plt.plot(range(len(test_H)), test_H)
            plt.savefig('.\\all-step\\test_prediction\\mod_' + str(mod) + '_No_' + str(i) + '_H_test_predict.jpg')
            plt.show()


    elif (real_value == 1):
        # =================测试训练集=====================================================
        # ===============训练集测试方法为拿第一个点作为已知S_O，根据公式计算后面的值=======================
        for i in range(len(v_state_train)):
            train_V = v_state_train[i]
            #             train_H = h_state_train[i]
            train_predict_V = np.zeros(len(train_V)).tolist()
            #             train_predict_H = np.zeros(len(train_H)).tolist()
            train_predict_V[0] = train_V[0]
            #             train_predict_H[0] = train_H[0]

            print("predict No. " + str(i) + " train dataset.")
            try:
                for j in range(1, len(train_V)):
                    #                     print(type(an_V))
                    if (mod == 0):
                        train_predict_V[j], _, _ = prediction(train_V[j - 1], an_V[i], b_V, sigma_B_V, 1, j)
                        p = 1
                        # train_predict_H[j] = prediction(train_predict_H[j - 1], an_H, b_H, sigma_B_H, 1, j)
                    if (mod == 1):
                        train_predict_V[j], _, _ = prediction_1(train_V[j - 1], an_V[i], b_V, sigma_B_V, 1, j)
                        # train_predict_H[j] = prediction_1(train_predict_H[j - 1], an_H, b_H, sigma_B_H, 1, j )
                    if (mod == 2):
                        train_predict_V[j], _, _ = prediction_2(train_V[j - 1], an_V[i], b_V, sigma_B_V, 1, j)

                #                 for j in range(1, len(train_H)):

                # if (mod == 0): # train_predict_V[j] = prediction(train_predict_V[j - 1], an_V, b_V, sigma_B_V, 1,
                # j) train_predict_H[j], _, _ = prediction(train_H[j - 1], an_H[i], b_H, sigma_B_H, 1, j) if (mod ==
                # 1): # train_predict_V[j] = prediction_1(train_predict_V[j - 1], an_V, b_V, sigma_B_H, 1,
                # j) train_predict_H[j], _, _ = prediction_1(train_H[j - 1], an_H[i], b_H, sigma_B_H, 1, j) if (mod
                # == 2): # train_predict_V[j] = prediction_2(train_predict_V[j - 1], an_H, b_H, sigma_B_H, 1,
                # j) train_predict_H[j], _, _ = prediction_2(train_H[j - 1], an_H[i], b_H, sigma_B_H, 1, j)
                V_TR.append(train_predict_V)
            #                 H_TR.append(train_predict_H)
            except Exception as e:
                print("No. " + str(i) + " train dataset get error when prediction.")
        # plt.plot(range(len(train_V)), train_predict_V) plt.plot(range(len(train_V)), train_V) plt.savefig(
        # '.\\one-step\\train_prediction\\mod_' + str(mod) + '_No_' + str(i) + '_V_train_predict.jpg') plt.show()
        # plt.plot(range(len(train_H)), train_predict_H) plt.plot(range(len(train_H)), train_H) plt.savefig(
        # '.\\one-step\\train_prediction\\mod_' + str(mod) + '_No_' + str(i) + '_H_train_predict.jpg') plt.show()

        # =================测试测试集=====================================================
        # ================测试集测试方法为最后一个训练集的点作为已知，根据公式计算测试集的值=========
        for i in range(len(v_state_train)):
            train_V = np.array(v_state_train[i])
            #             train_H = np.array(h_state_train[i])
            test_V = np.array(v_state_test[i])
            #             test_H = np.array(h_state_test[i])
            data_V = np.concatenate([np.array(v_state_train[i]), np.array(v_state_test[i])])
            #             data_H = np.concatenate([np.array(h_state_train[i]), np.array(h_state_test[i])])
            test_V_pie = np.array(v_state_test[i])
            #             test_H_pie = np.array(h_state_test[i])
            omega_V = np.zeros(len(test_V) + 1).tolist()
            error_V = np.zeros(len(test_V) + 1).tolist()
            #             omega_H = np.zeros(len(test_H)+1).tolist()
            #             error_H = np.zeros(len(test_H)+1).tolist()
            test_predict_V = copy.deepcopy(data_V[len(train_V) - 1:])
            #             test_predict_H = copy.deepcopy(data_H[len(train_H) - 1:])
            test_predict_V_pie = copy.deepcopy(data_V[len(train_V) - 1:])
            #             test_predict_H_pie = copy.deepcopy(data_H[len(train_H) - 1:])
            # test_predict_V = np.concatenate([np.array(v_state_train[i][-1]),np.array(v_state_test[i])])
            # test_predict_H = np.concatenate([np.array(h_state_train[i][-1]),np.array(h_state_test[i])])

            # test_predict_V[0] = test_V_pie[0]
            # test_predict_H[0] = test_H_pie[0]
            # b_V,an_V,sigma_B_V,_ = parameter_estimation(train_V,mod,b_range,b_granularity)
            # b_H, an_H, sigma_B_H, _ = parameter_estimation(train_H,mod,b_range,b_granularity)
            print("predict No. " + str(i) + " test dataset")
            #             try:
            for j in range(1, len(test_V) + 1):

                if (mod == 0):
                    test_predict_V[j], omega_V[j], error_V[j] = prediction(test_predict_V_pie[j - 1], an_V[i], b_V,
                                                                           sigma_B_V, 1, j + len(train_V))
                    # test_predict_H[j] = prediction(test_predict_H[j - 1], an_H, b_H, sigma_B_H, 1, j)
                if (mod == 1):
                    test_predict_V[j], omega_V[j], error_V[j] = prediction_1(test_predict_V_pie[j - 1], an_V[i], b_V,
                                                                             sigma_B_V, 1,
                                                                             j + len(train_V))
                    # test_predict_H[j] = prediction_1(test_predict_H[j - 1], an_H, b_H, sigma_B_H, 1, j )
                if (mod == 2):
                    #                     print(type(test_predict_V_pie))
                    #                     print(type(an_V))
                    test_predict_V[j], omega_V[j], error_V[j] = prediction_2(test_predict_V_pie[j - 1], an_V[i], b_V,
                                                                             sigma_B_V, 1,
                                                                             j + len(train_V))
                    # test_predict_H[j] = prediction_2(test_predict_H[j - 1], an_H, b_H, sigma_B_H, 1, j )
            #             for j in range(1, len(test_H) + 1):

            # if (mod == 0): # test_predict_V[j] = prediction(test_predict_V[j - 1], an_V, b_V, sigma_B_V, 1,
            # j) test_predict_H[j], omega_H[j], error_H[j] = prediction(test_predict_H_pie[j - 1], an_H[i], b_H,
            # sigma_B_H, 1, j + len(train_H)) if (mod == 1): # test_predict_V[j] = prediction_1(test_predict_V[j -
            # 1], an_V, b_V, sigma_B_V, 1, j) test_predict_H[j], omega_H[j], error_H[j] = prediction_1(
            # test_predict_H_pie[j - 1], an_H[i], b_H, sigma_B_H, 1, j + len(train_H)) if (mod == 2): #
            # test_predict_V[j] = prediction_2(test_predict_V[j - 1], an_V, b_V, sigma_B_V, 1, j) test_predict_H[j],
            # omega_H[j], error_H[j] = prediction_2(test_predict_H_pie[j - 1], an_H[i], b_H, sigma_B_H, 1,
            # j + len(train_H))
            V_TE.append(test_predict_V)
            #             H_TE.append(test_predict_H)
            V_omegas.append(omega_V)
            V_errors.append(error_V)
    #             H_omegas.append(omega_H)
    #             H_errors.append(error_H)
    #             except Exception as e:
    #                 print("No. " + str(i) + " train dataset get error when prediction.")
    #             plt.plot(range(len(test_V)), test_predict_V[1:])
    #             plt.plot(range(len(test_V)), test_V)
    #             plt.savefig('.\\one-step\\test_prediction\\mod_' + str(mod) + '_No_' + str(i) + '_V_test_predict.jpg')
    #             plt.show()
    #             plt.plot(range(len(test_H)), test_predict_H[1:])
    #             plt.plot(range(len(test_H)), test_H)
    #             plt.savefig('.\\one-step\\test_prediction\\mod_' + str(mod) + '_No_' + str(i) + '_H_test_predict.jpg')
    #             plt.show()

    #     return V_TR,H_TR,V_TE,H_TE, V_omegas, V_errors, H_omegas, H_errors
    return V_TR, V_TE, V_omegas, V_errors


def log_init(log_file):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    if log_file is None:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
    else:
        logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)


mod = 2
usedeep = ''
errtm = True
dataset = 'NASA'
random.seed(2)

data_folder = 'CMAPSSData' if dataset == 'NASA' else 'XJTU-SY_Bearing_Datasets'
folder = os.path.join(data_folder, 'Train_Test')
v_state_train = load_data(folder + '/v_state_train')
v_state_test = load_data(folder + '/v_state_test')
v_vibr_train = load_data(folder + '/v_vibr_train')
v_vibr_test = load_data(folder + '/v_vibr_test')
if dataset == 'XJTU':
    h_state_train = load_data(folder + '/h_state_train')
    h_state_test = load_data(folder + '/h_state_test')
    h_vibr_train = load_data(folder + '/h_vibr_train')
    h_vibr_test = load_data(folder + '/h_vibr_test')

folder_path = f'Results/{dataset}/wiener/jili_mod{mod}_usedeep{usedeep}_errtm{errtm}'
mkdir(folder_path)
log_file = os.path.join(folder_path, 'logging.txt')
log_init(log_file)
train_time_h = 0
test_time_h = 0

if dataset == "XJTU":
    deep_alpha_file = "./Results/XJTU/Deep_Wiener/datav_log/dw_wiener_alpinstate_alptypegru_alprnn@32@32@32@32_" \
                      "alpmlp@16@16@16_alpbidFalse_alpnhead1_omginstate_omginitb1_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_rc0/alpha"
    deep_b_file = "./Results/XJTU/Deep_Wiener/datav_log/dw_wiener_alpinstate_alptypegru_alprnn@32@32@32@32_" \
                  "alpmlp@16@16@16_alpbidFalse_alpnhead1_omginstate_omginitb1_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_rc0/b"
    deep_error_file = "./Results/XJTU/Deep_Wiener/datav_log/dw_wiener_alpinstate_alptypegru_alprnn@32@32@32@32_" \
                      "alpmlp@16@16@16_alpbidFalse_alpnhead1_omginstate_omginitb1_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_rc0/error"
    deep_omega_file = "./Results/XJTU/Deep_Wiener/datav_log/dw_wiener_alpinstate_alptypegru_alprnn@32@32@32@32_" \
                      "alpmlp@16@16@16_alpbidFalse_alpnhead1_omginstate_omginitb1_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_rc0/omega"
    h_deep_alpha_file = "Results/XJTU/Deep_Wiener/datah_log/dw_wiener_alpinstate_alptypegru_alprnn@32@32@32@32_alpmlp@16@16@16_alpbidFalse_alpnhead1_omginstate_omginitb1_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_rc0/alpha"
    h_deep_b_file = "./Results/XJTU/Deep_Wiener/datah_log/dw_wiener_alpinstate_alptypegru_alprnn@32@32@32@32_alpmlp@16@16@16_alpbidFalse_alpnhead1_omginstate_omginitb1_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_rc0/b"
    h_deep_error_file = "./Results/XJTU/Deep_Wiener/datah_log/dw_wiener_alpinstate_alptypegru_alprnn@32@32@32@32_alpmlp@16@16@16_alpbidFalse_alpnhead1_omginstate_omginitb1_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_rc0/error"
    h_deep_omega_file = "./Results/XJTU/Deep_Wiener/datah_log/dw_wiener_alpinstate_alptypegru_alprnn@32@32@32@32_alpmlp@16@16@16_alpbidFalse_alpnhead1_omginstate_omginitb1_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_rc0/omega"
    deep_alpha = load_data(deep_alpha_file)
    deep_b_value = load_data(deep_b_file).item()
    deep_error = load_data(deep_error_file)
    deep_omega = load_data(deep_omega_file)
    h_deep_alpha = load_data(deep_alpha_file)
    h_deep_b_value = load_data(deep_b_file).item()
    h_deep_error = load_data(deep_error_file)
    h_deep_omega = load_data(deep_omega_file)
    deep_alpha_list = [x.tolist()[0] for x in deep_alpha]
    deep_error_list = [np.array(tensor.tolist()).flatten() for tensor in deep_error]
    deep_omega_list = [np.array(tensor.tolist()).flatten() for tensor in deep_omega]
    h_deep_alpha_list = [x.tolist()[0] for x in h_deep_alpha]
    h_deep_error_list = [np.array(tensor.tolist()).flatten() for tensor in h_deep_error]
    h_deep_omega_list = [np.array(tensor.tolist()).flatten() for tensor in h_deep_omega]
else:
    deep_alpha_file = "./Results/NASA/Deep_Wiener/datav_log/alpinstate_alptypegru_alprnn@32@32@32@32_alpmlp@16@16@16_" \
                      "alpbidFalse_alpnhead1_omginstate_omginitb0.01_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_" \
                      "lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_rc0/alpha"
    deep_b_file = "./Results/NASA/Deep_Wiener/datav_log/alpinstate_alptypegru_alprnn@32@32@32@32_alpmlp@16@16@16_" \
                  "alpbidFalse_alpnhead1_omginstate_omginitb0.01_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_" \
                  "lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_rc0/b"
    deep_error_file = "./Results/NASA/Deep_Wiener/datav_log/alpinstate_alptypegru_alprnn@32@32@32@32_alpmlp@16@16@16_" \
                      "alpbidFalse_alpnhead1_omginstate_omginitb0.01_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_" \
                      "lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_rc0/error"
    deep_omega_file = "./Results/NASA/Deep_Wiener/datav_log/alpinstate_alptypegru_alprnn@32@32@32@32_alpmlp@16@16@16_" \
                      "alpbidFalse_alpnhead1_omginstate_omginitb0.01_errmlp@16@16_errOnormTrue_lrdecaysteplr_lrds20_" \
                      "lrdr0.9_lr0.002_kfold1_lamd10_normTrue_shufTrue_gradclipTrue@1_errtmTrue_cvTrue_rc0/omega"

    deep_alpha = load_data(deep_alpha_file)
    deep_b_value = load_data(deep_b_file).item()
    deep_error = load_data(deep_error_file)
    deep_omega = load_data(deep_omega_file)
    deep_alpha_list = [x.tolist()[0] for x in deep_alpha]
    deep_error_list = [np.array(tensor.tolist()).flatten() for tensor in deep_error]
    deep_omega_list = [np.array(tensor.tolist()).flatten() for tensor in deep_omega]

if dataset == 'XJTU':
    time_start = time.time()
    b_V_mod3, an_V_mod3, sigma_B_V_mod3, _ = parameter_estimation(v_state_train, mod, range(10000, 160000, 10000),
                                                                  100000)
    time_end = time.time()
    train_time_v = time_end - time_start
    if not errtm:
        sigma_B_V_mod3 = 0
    if usedeep == 'a':
        an_V_mod3 = deep_alpha_list
    elif usedeep == 'b':
        b_V_mod3 = deep_b_value
    # b_V_mod3, an_V_mod3, sigma_B_V_mod3, _ = parameter_estimation(v_state_train, 2, range(8000,8501),100000)
    time_start = time.time()
    V_TR_mod3, V_TE_mod3, \
    V_omegas_mod3, V_errors_mod3 = main(v_state_train, v_state_test, b_V_mod3, an_V_mod3, sigma_B_V_mod3, 2, 1)
    time_end = time.time()
    test_time_v = time_end - time_start

    time_start = time.time()
    b_H_mod3, an_H_mod3, sigma_B_H_mod3, _ = parameter_estimation(h_state_train, mod, range(10000, 160000, 10000),
                                                                  100000)
    if usedeep == 'a':
        an_H_mod3 = h_deep_alpha_list
    elif usedeep == 'b':
        b_H_mod3 = h_deep_b_value
    time_end = time.time()
    train_time_h = time_end - time_start
    if not errtm:
        sigma_B_H_mod3 = 0
    # b_H_mod3, an_H_mod3, sigma_B_H_mod3, _ = parameter_estimation(h_state_train, 2, range(8000,8501),100000)
    time_start = time.time()
    H_TR_mod3, H_TE_mod3, \
    H_omegas_mod3, H_errors_mod3 = main(h_state_train, h_state_test, b_H_mod3, an_H_mod3, sigma_B_H_mod3, 2, 1)
    time_end = time.time()
    test_time_h = time_end - time_start

elif dataset == 'NASA':
    time_start = time.time()
    b_V_mod3, an_V_mod3, sigma_B_V_mod3, _ = parameter_estimation(v_state_train, 2, range(100, 15000, 100), 10000)
    # print(b_V_mod3)
    time_end = time.time()
    train_time_v = time_end - time_start
    if not errtm:
        sigma_B_V_mod3 = 0
    time_start = time.time()
    V_TR_mod3, V_TE_mod3, \
    V_omegas_mod3, V_errors_mod3 = main(v_state_train, v_state_test, b_V_mod3, an_V_mod3, sigma_B_V_mod3, 2, 1)
    # b_H_mod3, an_H_mod3, sigma_B_H_mod3, _ = parameter_estimation(v_state_train, 0, range(8000,8501),10000)
    if usedeep == 'a':
        an_V_mod3 = deep_alpha_list
    elif usedeep == 'b':
        b_V_mod3 = deep_b_value
    time_end = time.time()
    test_time_v = time_end - time_start
    sigma_B_H_mod3 = None
    b_H_mod3 = None
    an_H_mod3 = None
    H_errors_mod3 = None
    H_omegas_mod3 = None
    H_TR_mod3 = None
    H_TE_mod3 = None

logging.info(f'V_TRAIN_TIME: {train_time_v}, V_TEST_TIME: {test_time_v}, '
             f'H_TRAIN_TIME: {train_time_h}, H_TEST_TIME: {test_time_h}')

# 定义要创建的目录
# 调用函数

plot_file = os.path.join(folder_path, 'plot_file.png')
log_file = os.path.join(folder_path, 'logging.txt')
test_res = os.path.join(folder_path, 'test_res.txt')
train_plot_file = os.path.join(folder_path, 'train_plot_file.png')
# train_log_file = os.path.join(folder_path, 'logging.txt')
train_res = os.path.join(folder_path, 'train_res.txt')
omega_pickles = os.path.join(folder_path, 'omega')
error_pickles = os.path.join(folder_path, 'error')
alpha_pickles = os.path.join(folder_path, 'alpha')
b_pickles = os.path.join(folder_path, 'b')
train_pickles = os.path.join(folder_path, 'train_est')
test_pickles = os.path.join(folder_path, 'test_est')
sigmaB_pickles = os.path.join(folder_path, 'sigmaB')

sigmaBs = {'v': sigma_B_V_mod3, 'h': sigma_B_H_mod3}
parambs = {'v': b_V_mod3, 'h': b_H_mod3}
alphas = {'v': an_V_mod3, 'h': an_H_mod3}
errors = {'v': V_errors_mod3, 'h': H_errors_mod3}
omegas = {'v': V_omegas_mod3, 'h': H_omegas_mod3}
train_ests = {'v': V_TR_mod3, 'h': H_TR_mod3}
test_ests = {'v': V_TE_mod3, 'h': H_TE_mod3}
save_data(sigmaBs, sigmaB_pickles)
save_data(omegas, omega_pickles)
save_data(alphas, alpha_pickles)
save_data(errors, error_pickles)
save_data(parambs, b_pickles)
save_data(train_ests, train_pickles)
save_data(test_ests, test_pickles)
