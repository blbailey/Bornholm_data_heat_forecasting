__author__ = 'Li Bai'

import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize, Bounds
from scipy.optimize import root
import statsmodels.tsa.seasonal
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.tsa.api import acf, pacf, graphics
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.colors as mcolors
import matplotlib as mpl

# this eneds to be speificed!!!
HEAT_MAX_REAL = 14.276114
HEAT_MIN_REAL = 0.09502
HEAT_MAX_APPRO = 14.417925
HEAT_MIN_APPRO = -0.04679


def pdf_Yeo(x, x_mu, sigma, lam):
    if (lam != 0) & (x_mu >= 0) & (x >= 0):
        mu = ((x_mu + 1) ** lam - 1) / lam
        x_tran = ((x + 1) ** lam - 1) / lam
        pdf_x = np.exp(-0.5 * (x_tran - mu) ** 2 / (
                sigma ** 2)) / np.sqrt(
            2 * np.pi * sigma ** 2) * ((x + 1) ** (lam - 1)) / (HEAT_MAX_APPRO - HEAT_MIN_APPRO)
    if (lam == 0) & (x_mu >= 0) & (x >= 0):
        mu = np.log(x_mu + 1)
        x_tran = np.log(x + 1)
        pdf_x = np.exp(-0.5 * (x_tran - mu) ** 2 / (
                sigma ** 2)) / np.sqrt(
            2 * np.pi * sigma ** 2) * (1 / (x + 1)) / (HEAT_MAX_APPRO - HEAT_MIN_APPRO)

    if (lam != 2) & (x_mu < 0) & (x < 0):
        mu = -((-x_mu + 1) ** (2 - lam) - 1) / (2 - lam)
        x_tran = -((-x + 1) ** (2 - lam) - 1) / (2 - lam)
        pdf_x = np.exp(-0.5 * (x_tran - mu) ** 2 / (
                sigma ** 2)) / np.sqrt(
            2 * np.pi * sigma ** 2) * (1 - x) ** (1 - lam) / (HEAT_MAX_APPRO - HEAT_MIN_APPRO)

    if (lam == 2) & (x_mu < 0) & (x < 0):
        mu = -np.log(-x_mu + 1)
        x_tran = -np.log(-x + 1)
        pdf_x = np.exp(-0.5 * (x_tran - mu) ** 2 / (
                sigma ** 2)) / np.sqrt(
            2 * np.pi * sigma ** 2) / (1 - x) / (HEAT_MAX_APPRO - HEAT_MIN_APPRO)

    return pdf_x


def pdf_general(x, x_mu, sigma, nu, col):
    # input x is the real value, and it must be normalized
    if col in ['OLS', 'RLS', 'RML']:
        return np.exp(-0.5 * (x - x_mu) ** 2 / (sigma ** 2)) / np.sqrt(
            2 * np.pi * sigma ** 2) / (HEAT_MAX_APPRO - HEAT_MIN_APPRO)

    if col in ['OLS_LG', 'RLS_LG', 'RML_LG']:
        # if x<=0:
        #     x=0.0001
        return np.exp(-0.5 * (log_transform(x, nu) - log_transform(x_mu, nu)) ** 2 / (
                sigma ** 2)) / np.sqrt(
            2 * np.pi * sigma ** 2) / (HEAT_MAX_APPRO - HEAT_MIN_APPRO) * nu / (x * (1 - (x) ** nu))
    if col in ['OLS_Yeo', 'RLS_Yeo', 'RML_Yeo']:
        return pdf_Yeo(x, x_mu, sigma, nu)


def cdf_empirical_general(x_real, x_mu, sigma, nu, col):
    # x_mu=0.01648; sigma=np.sqrt(0.0552); nu=0.1;
    # the real value of x_vals;
    gap = 0.0001
    x_vals = np.arange(0, HEAT_MAX_REAL, gap)
    x_vals = (x_vals - HEAT_MIN_APPRO) / (HEAT_MAX_APPRO - HEAT_MIN_APPRO)

    pdf_x = [];
    for x_val in x_vals:
        pdf_x.append(pdf_general(x_val, x_mu, sigma, nu, col))
    pdf_x = np.array(pdf_x)
    cdf_x = np.array([np.sum(pdf_x[0:k]) * gap for k in range(1, pdf_x.shape[
        0] + 1)])

    # x_real here is supposed to be in the original domain instead of (0,1) while the input is limited to 0,1
    x_real_origin = x_real * (HEAT_MAX_APPRO - HEAT_MIN_APPRO) + HEAT_MIN_APPRO
    x_real_loc = int(x_real_origin / gap + 1)
    crps_point = np.sum((cdf_x[0:x_real_loc]) ** 2) * gap + np.sum((1 - cdf_x[
                                                                        x_real_loc:]) ** 2) * gap

    return crps_point


def crps_general_array(x_reals, x_mus, sigma2s, nu, col):
    crps_arr = []

    x_reals = (x_reals - HEAT_MIN_APPRO) / (HEAT_MAX_APPRO - HEAT_MIN_APPRO)
    x_mus = (x_mus - HEAT_MIN_APPRO) / (HEAT_MAX_APPRO - HEAT_MIN_APPRO)
    for k in range(x_mus.shape[0]):
        crps_point = cdf_empirical_general(x_reals[k], x_mus[k], np.sqrt(
            sigma2s[k]), nu, col)
        crps_arr.append(crps_point)
    crps_arr = np.array(crps_arr)
    return crps_arr, crps_arr.mean(), crps_arr.reshape(-1, 24).mean(axis=0)


def log_transform(arr, nu):
    return np.log((arr ** nu) / (1 - (arr ** nu)))


def log_transform_inv(arr, nu):
    return (1 + 1 / np.exp(arr)) ** (-1 / nu)


def Yeo_transform(y, lam):
    if (lam != 0) & (y >= 0):
        return ((y + 1) ** lam - 1) / lam
    if (lam == 0) & (y >= 0):
        return np.log(y + 1)
    if (lam != 2) & (y < 0):
        return -((-y + 1) ** (2 - lam) - 1) / (2 - lam)
    if (lam == 2) & (y < 0):
        return -np.log(-y + 1)


def Yeo_transform_arr(arr, lam):
    return np.array([Yeo_transform(y, lam) for y in arr])


def Yeo_transform_inverse(x, lam):
    if lam == 0:
        if x >= 0:
            return np.exp(x) - 1
        else:
            return 1 - np.sqrt(1 - 2 * x)
    elif lam == 2:
        if x >= 0:
            return np.exp((lam * x + 1) / lam) - 1
        else:
            return 1 - np.exp(-x)
    else:
        if x >= 0:
            return (lam * x + 1) ** (1 / lam) - 1
        else:
            return 1 - (1 - x * (2 - lam)) ** (2 - lam)


def Yeo_transform_inverse_arr(arr, lam):
    return np.array([Yeo_transform_inverse(y, lam) for y in arr])


def OLS(df_features, df_output, N_train):
    df_features_train = df_features.to_numpy()[0:N_train, :]
    df_output_train = df_output.to_numpy()[0:N_train, :]
    df_features_test = df_features.to_numpy()[N_train:, :]
    df_output_test = df_output.to_numpy()[N_train:, :]

    Cov = df_features_train.T.dot(df_features_train);
    Covxy = df_features_train.T.dot(df_output_train);

    w_fix = np.linalg.inv(Cov).dot(Covxy)
    y_ols_test = w_fix.T.dot(df_features_test.T)[0, :]
    y_ols_test[y_ols_test < 0] = 0.
    y_ols_test[y_ols_test > 1] = 1.
    y_ols_train = w_fix.T.dot(df_features_train.T)[0, :]
    y_ols_train[y_ols_train < 0] = 0.
    y_ols_train[y_ols_train > 1] = 1.

    sigma2s_one = np.var(df_output_train - y_ols_test)
    sigma2s = np.ones(df_output.shape[0]) * sigma2s_one

    return y_ols_test, y_ols_train, sigma2s


def OLS_Yeo(df_features, df_output, N_train, nu):
    cols_tran = [
        'heat-lag-24',
        'heat-lag-25',
        # 'heat-t-23'
    ]
    df_features1 = df_features.copy()
    df_output1 = df_output.copy()
    for col in cols_tran:
        df_features1[col] = Yeo_transform_arr(df_features1[col].to_numpy(), nu)
    df_output1.iloc[:, 0] = Yeo_transform_arr(df_output1.to_numpy(), nu)

    df_features1 = df_features1.to_numpy()
    df_output1 = df_output1.to_numpy()

    df_features_train = df_features1[0:N_train, :]
    df_output_train = df_output1[0:N_train, :]
    df_features_test = df_features1[N_train:, :]
    df_output_test = df_output1[N_train:, :]

    Cov = df_features_train.T.dot(df_features_train);
    Covxy = df_features_train.T.dot(df_output_train);

    w_fix = np.linalg.inv(Cov).dot(Covxy)
    y_ols_test = w_fix.T.dot(df_features_test.T)[0, :]
    y_ols_test = Yeo_transform_inverse_arr(y_ols_test, nu)
    y_ols_test[y_ols_test < 0] = 0.
    y_ols_test[y_ols_test > 1] = 1.

    y_ols_train = w_fix.T.dot(df_features_train.T)[0, :]

    sigma = np.mean((df_output_train - y_ols_train) ** 2)
    sigma2s = np.array([sigma] * df_features.shape[0])

    y_ols_train = Yeo_transform_inverse_arr(y_ols_train, nu)
    y_ols_train[y_ols_train < 0] = 0.
    y_ols_train[y_ols_train > 1] = 1.

    return y_ols_test, y_ols_train, sigma2s


def OLS_LG(df_features, df_output, N_train, nu):
    # use the normalized way to normalized it back!!!????
    # for more data, the max is increased to 0.126, the UB=HEAT_MAX
    df_features1 = df_features.copy()
    df_output1 = df_output.copy()

    # ganrantee that all the inputs are going to be forecasted!
    cols_tran = [
        'heat-lag-24',
        'heat-lag-25',
        # 'heat-t-23'
    ]

    for col in cols_tran:
        df_features1[col] = log_transform(df_features1[col].to_numpy(),
                                          nu);

    df_output1.iloc[:, 0] = log_transform(df_output1.to_numpy(),
                                          nu);

    df_features1 = df_features1.to_numpy()
    df_output1 = df_output1.to_numpy()

    df_features_train = df_features1[0:N_train, :]
    df_output_train = df_output1[0:N_train, :]
    df_features_test = df_features1[N_train:, :]
    df_output_test = df_output1[N_train:, :]

    Cov = df_features_train.T.dot(df_features_train);
    Covxy = df_features_train.T.dot(df_output_train);

    w_fix = np.linalg.inv(Cov).dot(Covxy)
    y_ols_test = w_fix.T.dot(df_features_test.T)[0, :]
    y_ols_test = log_transform_inv(y_ols_test, nu)
    y_ols_test[y_ols_test < 0] = 0.
    y_ols_test[y_ols_test > 1] = 1.
    y_ols_test = y_ols_test

    y_ols_train = w_fix.T.dot(df_features_train.T)[0, :]
    sigma = np.mean((df_output_train - y_ols_train) ** 2)
    sigma2s = np.array([sigma] * df_features.shape[0])

    y_ols_train = log_transform_inv(y_ols_train, nu)
    y_ols_train[y_ols_train < 0] = 0.
    y_ols_train[y_ols_train > 1] = 1.
    y_ols_train = y_ols_train

    return y_ols_test, y_ols_train, sigma2s


def RLS(df_features, df_output, N_train, lam_fgt):
    # R is a matrix
    df_features1 = df_features.to_numpy();
    df_output1 = df_output.to_numpy()
    num_beta = df_features1.shape[1]

    w_n_1 = np.zeros(shape=(num_beta, 1))
    R_n_1 = np.identity(n=num_beta) * 10
    sigma2_n_1 = 1

    import math
    y_news = [];
    y_olds = []
    ws = []
    ws_day = []
    sigma2s = []
    # Chapter9 from S.Haykin - Adaptive Filtering Theory - Prentice Hall, 2002.
    for k in range(df_features1.shape[0]):
        u_n = df_features1[k, :].reshape(-1, 1)
        d_n = df_output1[k, :]
        ws.append(w_n_1[:, 0])
        sigma2s.append(sigma2_n_1)

        if math.fmod(k, 24) == 0:
            groupk = int(k / 24)
            u_groupk = df_features1[groupk * 24:(groupk + 1) * 24, :]
            d_groupk = df_output1[groupk * 24:(groupk + 1) * 24, :]
            #
            y_news = y_news + w_n_1.T.dot(u_groupk.T)[0, :].tolist()
            y_olds = y_olds + (d_groupk[:, 0]).tolist()
            ws_day.append(w_n_1[:, 0])

        alf_n = d_n - u_n.T.dot(w_n_1)  # scalar
        R_n = lam_fgt * R_n_1 + u_n.dot(u_n.T)
        w_n = w_n_1 + np.linalg.inv(R_n).dot(u_n) * alf_n
        # R_n=lam_fgt*R_n_1+(1-lam_fgt)*2*u_n.dot(u_n.T)
        # w_n=w_n_1-(1-lam_fgt)*2*np.linalg.inv(R_n).dot(-u_n)*alf_n

        y_fore_n = u_n.T.dot(w_n)[0, 0];
        if y_fore_n > 1:
            y_fore_n = 1
        elif y_fore_n < 0:
            y_fore_n = 0
        else:
            y_fore_n = y_fore_n

        w_sigma2_n = 4 * (y_fore_n) * (1 - y_fore_n)
        err_n = d_n - u_n.T.dot(w_n_1)  # shape is (1x1)

        lam_n_sigma2 = 1 - (1 - lam_fgt) * w_sigma2_n
        sigma2_n = lam_n_sigma2 * sigma2_n_1 + (1 - lam_n_sigma2) * (
            err_n[0, 0]) ** 2

        sigma2_n_1 = sigma2_n

        R_n_1 = R_n
        w_n_1 = w_n

    y_news = np.array(y_news)

    y_news[y_news <= 0] = 0.;
    y_news[y_news > 1] = 1.

    y_rls_test = y_news[N_train:]

    y_rls_train = y_news[0:N_train]

    return y_rls_test, y_rls_train, ws_day, sigma2s


def RLS_Yeo(df_features, df_output, N_train, lam_fgt, nu):
    cols_tran = [
        'heat-lag-24',
        'heat-lag-25',
        # 'heat-t-23'
    ]
    df_features1 = df_features.copy()
    df_output1 = df_output.copy()
    for col in cols_tran:
        df_features1[col] = Yeo_transform_arr(df_features1[col].to_numpy(), nu)
    df_output1.iloc[:, 0] = Yeo_transform_arr(df_output1.to_numpy(), nu)

    df_features1 = df_features1.to_numpy()
    df_output1 = df_output1.to_numpy()
    num_beta = df_features1.shape[1]

    w_n_1 = np.zeros(shape=(num_beta, 1))
    R_n_1 = np.identity(n=num_beta) * 10
    sigma2_n_1 = 0.

    import math
    y_news = [];
    y_olds = []
    ws = []
    ws_day = []

    sigma2s = []
    zeros = []
    # Chapter9 from S.Haykin - Adaptive Filtering Theory - Prentice Hall, 2002.
    # for k in range(df_features.shape[0]):
    for k in range(df_features.shape[0]):

        u_n = df_features1[k, :].reshape(-1, 1)
        d_n = df_output1[k, :]
        ws.append(w_n_1[:, 0])
        sigma2s.append(sigma2_n_1)

        if math.fmod(k, 24) == 0:
            groupk = int(k / 24)
            u_groupk = df_features1[groupk * 24:(groupk + 1) * 24, :]
            d_groupk = df_output1[groupk * 24:(groupk + 1) * 24, :]

            # sometimes if history value in the input features is -inf, it can give rise
            # to its output to be UB; inside we should be careful!!! if else:
            #
            y_new_grouk = w_n_1.T.dot(u_groupk.T)[0, :]
            y_news = y_news + y_new_grouk.tolist()

            y_olds = y_olds + (d_groupk[:, 0]).tolist()
            ws_day.append(w_n_1[:, 0])

        alf_n = d_n - u_n.T.dot(w_n_1)  # scalar
        #
        # R_n = lam_fgt * R_n_1 + (1 - lam_fgt) *2* u_n.dot(u_n.T)
        # w_n = w_n_1 - (1 - lam_fgt) *2* np.linalg.inv(R_n).dot(-u_n) * alf_n

        R_n = lam_fgt * R_n_1 + u_n.dot(u_n.T)
        w_n = w_n_1 - np.linalg.inv(R_n).dot(-u_n) * alf_n

        y_fore_n = u_n.T.dot(w_n)[0, 0];
        y_fore_n = Yeo_transform_inverse(y_fore_n, nu)
        # print(y_fore_n)

        if y_fore_n > 1:
            y_fore_n = 1
        elif y_fore_n < 0:
            y_fore_n = 0
        else:
            y_fore_n = y_fore_n

        w_sigma2_n = 4 * (y_fore_n) * (1 - y_fore_n)
        err_n = d_n - u_n.T.dot(w_n_1)  # shape is (1x1)

        lam_n_sigma2 = 1 - (1 - lam_fgt) * w_sigma2_n
        # print(sigma2_n_1)
        # print(lam_n_sigma2)
        # print(err_n[0,0])
        sigma2_n = lam_n_sigma2 * sigma2_n_1 + (1 - lam_n_sigma2) * (
            err_n[0, 0]) ** 2

        sigma2_n_1 = sigma2_n

        R_n_1 = R_n
        w_n_1 = w_n

    y_news = np.array(y_news)
    y_news = Yeo_transform_inverse_arr(y_news, nu)

    y_news[y_news <= 0] = 0.;
    y_news[y_news > 1] = 1.

    y_Yeo_rls_test = y_news[N_train:]
    y_Yeo_rls_train = y_news[0:N_train]

    return y_Yeo_rls_test, y_Yeo_rls_train, ws_day, sigma2s


def RLS_LG(df_features, df_output, N_train, lam_fgt, nu):
    # use the normalized way to normalized it back!!!????
    # for more data, the max is increased to 0.126, the UB=HEAT_MAX

    # ganrantee that all the inputs are going to be forecasted!
    cols_tran = [
        'heat-lag-24',
        'heat-lag-25',
        # 'heat-t-23'
    ]
    df_features1 = df_features.copy()
    df_output1 = df_output.copy()

    for col in cols_tran:
        df_features1[col] = log_transform(df_features1[col].to_numpy(),
                                          nu);

    df_output1.iloc[:, 0] = log_transform(df_output1.to_numpy(),
                                          nu);

    df_features1 = df_features1.to_numpy()
    df_output1 = df_output1.to_numpy()
    num_beta = df_features1.shape[1]
    # R is a matrix
    w_n_1 = np.zeros(shape=(num_beta, 1))
    R_n_1 = np.identity(n=num_beta) * 10
    sigma2_n_1 = 0.076

    import math
    y_news = [];
    y_olds = []
    ws = []
    ws_day = []

    sigma2s = []
    zeros = []
    # Chapter9 from S.Haykin - Adaptive Filtering Theory - Prentice Hall, 2002.
    # for k in range(df_features.shape[0]):
    for k in range(df_features1.shape[0]):

        u_n = df_features1[k, :].reshape(-1, 1)
        d_n = df_output1[k, :]
        ws.append(w_n_1[:, 0])
        sigma2s.append(sigma2_n_1)

        if math.fmod(k, 24) == 0:
            groupk = int(k / 24)
            u_groupk = df_features1[groupk * 24:(groupk + 1) * 24, :]
            d_groupk = df_output1[groupk * 24:(groupk + 1) * 24, :]

            # sometimes if history value in the input features is -inf, it can give rise
            # to its output to be UB; inside we should be careful!!! if else:
            #
            y_new_grouk = w_n_1.T.dot(u_groupk.T)[0, :]
            # y_new_grouk[y_new_grouk<LB_tran]==-np.inf
            y_news = y_news + y_new_grouk.tolist()

            y_olds = y_olds + (d_groupk[:, 0]).tolist()
            ws_day.append(w_n_1[:, 0])

        if (np.sum(d_n == -np.inf) == 0):

            alf_n = d_n - u_n.T.dot(w_n_1)  # scalar

            # R_n = lam_fgt * R_n_1 + (1 - lam_fgt) * 2 * u_n.dot(u_n.T)
            # w_n = w_n_1 - (1 - lam_fgt) * 2 * np.linalg.inv(R_n).dot(
            #     -u_n) * alf_n
            R_n = lam_fgt * R_n_1 + u_n.dot(u_n.T)
            w_n = w_n_1 - np.linalg.inv(R_n).dot(
                -u_n) * alf_n
            y_fore_n = u_n.T.dot(w_n)[0, 0];
            y_fore_n = log_transform_inv(y_fore_n, nu)
            if y_fore_n > 1:
                y_fore_n = 1
            elif y_fore_n < 0:
                y_fore_n = 0
            else:
                y_fore_n = y_fore_n

            w_sigma2_n = 4 * (y_fore_n) * (1 - y_fore_n)
            err_n = d_n - u_n.T.dot(w_n_1)  # shape is (1x1)

            lam_n_sigma2 = 1 - (1 - lam_fgt) * w_sigma2_n
            sigma2_n = lam_n_sigma2 * sigma2_n_1 + (1 - lam_n_sigma2) * (
                err_n[0, 0]) ** 2

            sigma2_n_1 = sigma2_n

            R_n_1 = R_n
            w_n_1 = w_n

    y_news = np.array(y_news)
    y_news = log_transform_inv(y_news, nu)
    y_news[y_news <= 0] = 0.;
    y_news[y_news > 1] = 1.

    y_news = y_news;

    y_rls_lg_test = y_news[N_train:]
    y_rls_lg_train = y_news[0:N_train]
    return y_rls_lg_test, y_rls_lg_train, ws_day, sigma2s


def RML(df_features, df_output, N_train, NUM, lam_fgt):
    df_features1 = df_features.to_numpy();
    df_output1 = df_output.to_numpy()
    # P is a matrix
    num_beta = df_features1.shape[1]
    delta_n_1 = 0.001 * np.ones(shape=(1, 1))  # delta=sigma**2
    sigma2_n_1 = 0.001
    w_n_1 = np.zeros(shape=(num_beta, 1))
    para_n_1 = np.concatenate((delta_n_1, w_n_1), axis=0)
    # R_n_1=np.zeros(shape=(num_beta+1,num_beta+1))
    R_n_1 = np.identity(num_beta + 1) / 100

    import math
    y_news = [];
    y_olds = []
    ws = []
    ws_day = []
    sigma2s = []

    # Chapter9 from S.Haykin - Adaptive Filtering Theory - Prentice Hall, 2002.
    for k in range(df_features1.shape[0]):
        u_n = df_features1[k, :].reshape(-1, 1)  # x
        d_n = df_output1[k, :]  # y
        ws.append(w_n_1[:, 0])  #
        sigma2s.append(delta_n_1[0, 0])

        if math.fmod(k, 24) == 0:
            groupk = int(k / 24)
            u_groupk = df_features1[groupk * 24:(groupk + 1) * 24, :]
            d_groupk = df_output1[groupk * 24:(groupk + 1) * 24, :]
            #
            y_news = y_news + w_n_1.T.dot(u_groupk.T)[0, :].tolist()
            y_olds = y_olds + (d_groupk[:, 0]).tolist()
            ws_day.append(w_n_1[:, 0])

        alf_n = d_n - u_n.T.dot(w_n_1)
        h_n_0 = -1 / 2 + 1 / 2 / (delta_n_1) * (alf_n) ** 2
        h_n = -1 / delta_n_1 * (-u_n) * alf_n
        h_n_all = np.concatenate((h_n_0, h_n))

        R_n = lam_fgt * R_n_1 + (1 - lam_fgt) * h_n_all.dot(h_n_all.T)
        # R_n_0=lam_fgt*R_n_1_0+(1-lam_fgt)*h_n_0*h_n_0

        R_n_1 = R_n
        # R_n_1_0 = R_n_0

        if k > NUM + num_beta:
            para_n = para_n_1 + (1 - lam_fgt) * np.linalg.inv(R_n).dot(h_n_all)
            para_n_1 = para_n

            w_n_1 = para_n[1:, :]
            delta_n_1 = np.exp(para_n[0:1, :])

    y_news = np.array(y_news)
    y_news[y_news <= 0] = 0.
    y_news[y_news >= 1] = 1.

    y_news_test = y_news[N_train:]

    y_news_train = y_news[0:N_train]

    return y_news_test, y_news_train, ws_day, sigma2s


def RML_Yeo(df_features, df_output, N_train, NUM, lam_fgt, nu):
    cols_tran = [
        'heat-lag-24',
        'heat-lag-25',
        # 'heat-t-23'
    ]
    df_features1 = df_features.copy()
    df_output1 = df_output.copy()
    for col in cols_tran:
        df_features1[col] = Yeo_transform_arr(df_features1[col].to_numpy(), nu)
    df_output1.iloc[:, 0] = Yeo_transform_arr(df_output1.to_numpy(), nu)

    df_features1 = df_features1.to_numpy()
    df_output1 = df_output1.to_numpy()

    num_beta = df_features1.shape[1]
    delta_n_1 = 0.001 * np.ones(shape=(1, 1))  # delta=sigma**2
    sigma2_n_1 = 0.001
    w_n_1 = np.zeros(shape=(num_beta, 1))
    para_n_1 = np.concatenate((delta_n_1, w_n_1), axis=0)
    # R_n_1=np.zeros(shape=(num_beta+1,num_beta+1))
    R_n_1 = np.identity(num_beta + 1) / 100

    import math
    y_news = [];
    y_olds = []
    ws = []
    ws_day = []
    sigma2s = []
    # Chapter9 from S.Haykin - Adaptive Filtering Theory - Prentice Hall, 2002.
    for k in range(df_features1.shape[0]):
        u_n = df_features1[k, :].reshape(-1, 1)  # x
        d_n = df_output1[k, :]  # y
        ws.append(w_n_1[:, 0])  #
        sigma2s.append(delta_n_1[0, 0])

        if math.fmod(k, 24) == 0:
            groupk = int(k / 24)
            u_groupk = df_features1[groupk * 24:(groupk + 1) * 24, :]
            d_groupk = df_output1[groupk * 24:(groupk + 1) * 24, :]
            #
            y_news = y_news + w_n_1.T.dot(u_groupk.T)[0, :].tolist()
            y_olds = y_olds + (d_groupk[:, 0]).tolist()
            ws_day.append(w_n_1[:, 0])

            # alf_n = d_n - u_n.T.dot(w_n_1)
            # h_n_0 = -1 / 2  + 1 / 2 / (delta_n_1) * (alf_n) ** 2
            # h_n = -1 / delta_n_1 * (-u_n) * alf_n
            # h_n_all = np.concatenate((h_n_0, h_n))
            #
            # R_n = lam_fgt * R_n_1 + (1 - lam_fgt) * h_n_all.dot(h_n_all.T)
            #
            # R_n_1 = R_n

        alf_n = d_n - u_n.T.dot(w_n_1)
        h_n_0 = -1 / 2 + 1 / 2 / (delta_n_1) * (alf_n) ** 2
        h_n = -1 / delta_n_1 * (-u_n) * alf_n
        h_n_all = np.concatenate((h_n_0, h_n))

        R_n = lam_fgt * R_n_1 + (1 - lam_fgt) * h_n_all.dot(h_n_all.T)

        R_n_1 = R_n

        if k > NUM + num_beta:
            para_n = para_n_1 + (1 - lam_fgt) * np.linalg.inv(R_n).dot(h_n_all)

            para_n_1 = para_n

            w_n_1 = para_n[1:, :]
            delta_n_1 = np.exp(para_n[0:1, :])

    y_news = np.array(y_news)
    y_news = Yeo_transform_inverse_arr(y_news, nu)

    y_news[y_news <= 0] = 0.;
    y_news[y_news > 1] = 1.

    y_rml_Yeo_test = y_news[N_train:]
    y_rml_Yeo_train = y_news[0:N_train]

    return y_rml_Yeo_test, y_rml_Yeo_train, ws_day, sigma2s


def RML_LG(df_features, df_output, N_train, NUM, lam_fgt, nu):
    # use the normalized way to normalized it back!!!????

    cols_tran = [
        'heat-lag-24',
        'heat-lag-25',
        # 'heat-t-23'
    ]
    df_features1 = df_features.copy()
    df_output1 = df_output.copy()

    for col in cols_tran:
        df_features1[col] = log_transform(
            df_features1[col].to_numpy(),
            nu);

    df_output1.iloc[:, 0] = log_transform(
        df_output1.to_numpy(),
        nu);

    df_features1 = df_features1.to_numpy()
    df_output1 = df_output1.to_numpy()

    # plt.figure();plt.plot(df_features_x_tran['temp-t'], df_output_y_tran[
    #     'heat'],linestyle='dotted')

    num_beta = df_features1.shape[1]
    delta_n_1 = 0.001 * np.ones(shape=(1, 1))  # delta=sigma**2
    sigma2_n_1 = 0.001
    w_n_1 = np.zeros(shape=(num_beta, 1))
    para_n_1 = np.concatenate((delta_n_1, w_n_1), axis=0)
    # R_n_1=np.zeros(shape=(num_beta+1,num_beta+1))
    R_n_1 = np.identity(num_beta + 1) / 100

    import math
    y_news = [];
    y_olds = []
    ws = []
    ws_day = []
    sigma2s = []
    # Chapter9 from S.Haykin - Adaptive Filtering Theory - Prentice Hall, 2002.
    for k in range(df_features1.shape[0]):
        u_n = df_features1[k, :].reshape(-1, 1)  # x
        d_n = df_output1[k, :]  # y
        ws.append(w_n_1[:, 0])  #
        sigma2s.append(delta_n_1[0, 0])

        if math.fmod(k, 24) == 0:
            groupk = int(k / 24)
            u_groupk = df_features1[groupk * 24:(groupk + 1) * 24, :]
            d_groupk = df_output1[groupk * 24:(groupk + 1) * 24, :]
            #
            y_new_grouk = w_n_1.T.dot(u_groupk.T)[0, :]
            # y_new_grouk[y_new_grouk<LB_tran]=-np.inf
            y_news = y_news + y_new_grouk.tolist()

            y_olds = y_olds + (d_groupk[:, 0]).tolist()
            ws_day.append(w_n_1[:, 0])
        if (np.sum(d_n == -np.inf) == 0):

            alf_n = d_n - u_n.T.dot(w_n_1)
            h_n_0 = -1 / 2 + 1 / 2 / (delta_n_1) * (alf_n) ** 2
            h_n = -1 / delta_n_1 * (-u_n) * alf_n
            h_n_all = np.concatenate((h_n_0, h_n))

            R_n = lam_fgt * R_n_1 + (1 - lam_fgt) * h_n_all.dot(h_n_all.T)

            R_n_1 = R_n

            if k > NUM + num_beta:
                para_n = para_n_1 + (1 - lam_fgt) * np.linalg.inv(R_n).dot(
                    h_n_all)

                para_n_1 = para_n

                w_n_1 = para_n[1:, :]
                delta_n_1 = np.exp(para_n[0:1, :])

    y_news = np.array(y_news)
    y_news = log_transform_inv(y_news, nu)
    y_news[y_news <= 0.] = 0.
    y_news[y_news >= 1.] = 1.
    y_news = y_news

    y_rml_lg_test = y_news[N_train:]
    y_rml_lg_train = y_news[0:N_train]

    return y_rml_lg_test, y_rml_lg_train, ws_day, sigma2s


def similar_day_select_only(df_features_train, df_features_test):
    """similar day selection is based on the weather of the current day and
    the yesterday; using clustering or other methods doing so: df_features
    are normalized features including ['temp-t', 'humid-t', 'DNI-t',
    'windspeed-t', 'heat-t-24', 'const'] """
    from sklearn.cluster import KMeans
    df_features_train1 = df_features_train.copy()
    df_features_train1['date'] = df_features_train.index.date
    df_features_train1_mean = df_features_train1.groupby('date').mean()

    df_features_test1 = df_features_test.copy()
    df_features_test1['date'] = df_features_test.index.date
    df_features_test1_mean = df_features_test1.groupby('date').mean()

    mdl_kmeans = KMeans(n_clusters=2).fit(
        df_features_train1_mean)
    cluster_test = mdl_kmeans.predict(df_features_test1_mean)
    cluster_train = mdl_kmeans.predict(df_features_train1_mean)

    df_cluster_train = pd.DataFrame(columns=['day'],
                                    index=df_features_train1.iloc[0::24, -1])

    df_cluster_test = pd.DataFrame(columns=['day'],
                                   index=df_features_test1.iloc[0::24, -1])
    df_cluster_train['day'] = cluster_train
    df_cluster_test['day'] = cluster_test

    df_output1_test = df_output_test.copy()
    df_output1_train = df_output_train.copy()

    df_output1_train['cluster'] = np.array(cluster_train.tolist() * 24).reshape(
        24, -1).T.reshape(-1, 1)[:, 0]

    df_output1_train_cluster0 = df_output1_train.loc[df_output1_train[
                                                         'cluster'] == 0, ['heat']]

    df_output1_train_cluster1 = df_output1_train.loc[df_output1_train[
                                                         'cluster'] == 1, ['heat']]

    df_output1_test['cluster'] = np.array(cluster_test.tolist() * 24).reshape(
        24, -1).T.reshape(-1, 1)[:, 0]

    return df_output1_train, df_output1_test


def train_empirical_prob(df_output_test_un, df_output_train_un):
    """yesterday model: variances are accumulated online based on yesterdays
    model; while how to calculate the variance which is hard"""
    """how about hourly model empirical things for hour! use what from the
    training dataset: do we update it based on history dataset???"""
    """persistence probability: yesterday model; how to calculate it based on yesterday..using online accumulated errors
    using a forgetting parameter as well; theoretical it only consider one
    example.. or 24 hours ago; 0.958!; """

    heat_train = df_output_train_un['heat-lag-0'].to_numpy()
    ecdf_train = ECDF(heat_train)
    gap = 0.0001
    x_vals = np.arange(0, HEAT_MAX_REAL, gap)
    cdf_x = ecdf_train(x_vals)
    crps_arr = []
    x_reals = df_output_test_un['heat-lag-0'].to_numpy()
    for k in range(x_reals.shape[0]):
        x_real = x_reals[k]
        x_real_loc = int(x_real / gap + 1)
        crps_point = np.sum(cdf_x[0:x_real_loc]) * gap + np.sum(1 - cdf_x[
                                                                    x_real_loc:]) * gap
        crps_arr.append(crps_point)
    crps_arr = np.array(crps_arr)
    return crps_arr, crps_arr.mean(), crps_arr.reshape(-1, 24).mean(axis=0)


def train_hour_prob(heat_train, heat_test):
    ecdf_train = ECDF(heat_train)
    gap = 0.0001
    x_vals = np.arange(0, HEAT_MAX_REAL, gap)
    cdf_x = ecdf_train(x_vals)
    crps_arr = []
    x_reals = heat_test
    for k in range(x_reals.shape[0]):
        x_real = x_reals[k]
        x_real_loc = int(x_real / gap + 1)
        crps_point = np.sum(cdf_x[0:x_real_loc]) * gap + np.sum(1 - cdf_x[
                                                                    x_real_loc:]) * gap
        crps_arr.append(crps_point)
    crps_arr = np.array(crps_arr)
    return crps_arr


def train_hour_probs(df_output_test_un, df_output_train_un):
    heat_trains = df_output_train_un['heat-lag-0'].to_numpy().reshape(-1, 24)
    heat_tests = df_output_test_un['heat-lag-0'].to_numpy().reshape(-1, 24)

    crps_tests = []
    for h in range(24):
        heat_train = heat_trains[:, h]
        heat_test = heat_tests[:, h]

        crps_arr = train_hour_prob(heat_train, heat_test)
        crps_tests.append(crps_arr)
    crps_tests = np.array(crps_tests).T
    return crps_tests.reshape(-1, 1)[:, 0], crps_tests.reshape(-1, 1).mean(), \
           crps_tests.mean(
               axis=0)


def similar_day_select_probs(df_features_train, df_features_test, df_output_train, df_output_test):
    """similar day selection is based on the weather of the current day and
    the yesterday; using clustering or other methods doing so: df_features
    are normalized features including ['temp-t', 'humid-t', 'DNI-t',
    'windspeed-t', 'heat-t-24', 'const'] """
    from sklearn.cluster import KMeans
    df_features_train1 = df_features_train.copy()
    df_features_train1['date'] = df_features_train.index.date
    df_features_train1_mean = df_features_train1.groupby('date').mean()

    df_features_test1 = df_features_test.copy()
    df_features_test1['date'] = df_features_test.index.date
    df_features_test1_mean = df_features_test1.groupby('date').mean()

    mdl_kmeans = KMeans(n_clusters=2).fit(
        df_features_train1_mean)
    cluster_test = mdl_kmeans.predict(df_features_test1_mean)
    cluster_train = mdl_kmeans.predict(df_features_train1_mean)

    df_cluster_train = pd.DataFrame(columns=['day'],
                                    index=df_features_train1.iloc[0::24, -1])

    df_cluster_test = pd.DataFrame(columns=['day'],
                                   index=df_features_test1.iloc[0::24, -1])
    df_cluster_train['day'] = cluster_train
    df_cluster_test['day'] = cluster_test

    df_output1_test = df_output_test.copy()
    df_output1_train = df_output_train.copy()

    df_output1_train['cluster'] = np.array(cluster_train.tolist() * 24).reshape(
        24, -1).T.reshape(-1, 1)[:, 0]

    df_output1_train_cluster0 = df_output1_train.loc[df_output1_train[
                                                         'cluster'] == 0, ['heat-lag-0']]

    df_output1_train_cluster1 = df_output1_train.loc[df_output1_train[
                                                         'cluster'] == 1, ['heat-lag-0']]

    df_output1_test['cluster'] = np.array(cluster_test.tolist() * 24).reshape(
        24, -1).T.reshape(-1, 1)[:, 0]

    crps_tests = []
    for idx in df_output1_test.index:
        if df_output1_test.loc[idx, 'cluster'] == 0:
            df_output1_train_h = df_output1_train_cluster0.loc[
                df_output1_train_cluster0.index.hour == idx.hour, ['heat-lag-0']]

        else:
            df_output1_train_h = df_output1_train_cluster1.loc[
                df_output1_train_cluster1.index.hour == idx.hour, ['heat-lag-0']]

        crps_arr = train_hour_prob(df_output1_train_h.to_numpy()[:, 0],
                                   df_output1_test.loc[idx, ['heat-lag-0']].to_numpy())
        crps_tests.append(crps_arr)

    crps_tests = np.array(crps_tests)
    return crps_tests, crps_tests.mean(), crps_tests.reshape(-1, 24).mean(
        axis=0)




