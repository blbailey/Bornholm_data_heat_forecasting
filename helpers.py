__author__ = 'Li Bai'
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import datetime

# shift
LAG_HEAT=1;  # heat
LAG_TMP=2; # temperature
LAG_HMD=2; # humidity
LAG_DNI=2; # solar radiance
LAG_WIND=2; # wind
LAG_DAY=2 # daily sine or consine curve
LAG_WEEK=2; # weekly sin or consine curve
SHIFT=0; # shift for variables except heat
SHIFT_HEAT=24; # shift for heat series by using the history data



# parameters for lags; for example, if lag_heat=2, generate new variables of heat time series {y_t}, {y_t-1} and {
# y_t-2}; if the time shift_heat=24 together with lag_heat 2, then new variables of {y_t-24}, {y_t-1-24} and {
#  y_t-2-24} will be generated. Likewise, for the others

#  weather related variables are derived from forecasts values, which can be obtained any lagged time step of t;
#  while for the heat load, only history data are available; so the shift time of the heat load is 24, indicating
#  yesterday



# dictionary (key= variable, value=lag steps)
LAG_DICT={'heat': LAG_HEAT, 'temperature':LAG_TMP, 'humidity':LAG_HMD,
          'DNI':LAG_DNI, 'windspeed':LAG_WIND,'Day sin':LAG_DAY,
          'Day cos':LAG_DAY, 'Week sin':LAG_WEEK, 'Week cos':LAG_WEEK}


# =====generate a daily data for them
LAG_HEAT1=23;LAG_TMP1=23; LAG_HMD1=23;LAG_DNI1=23;LAG_WIND1=23;LAG_DAY1=23;
LAG_WEEK1=23;SHIFT1=0;SHIFT_HEAT1=24;

LAG_DICT1={'heat': LAG_HEAT1, 'temperature':LAG_TMP1, 'humidity':LAG_HMD1,
          'DNI':LAG_DNI1, 'windspeed':LAG_WIND1, 'Day cos':LAG_DAY1}





def plot_acf_or_pacf(df_out):
    from statsmodels.tsa.api import acf, pacf, graphics
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    for key in df_out.keys():
        plot_acf(df_out[key], lags=100, title="Autocorrelation: "+key)
        # plot_pacf(df_out[key], lags=40, title="PACF: "+key)
        plt.xlabel('Lag (hour)')


def fft_analy(df_out):
    import tensorflow as tf
    day = 24*60*60
    week=7*day
    # month=30*day
    year = (365.2425)*day

    # dx=[0, 5, 8, 11, 14, 17]

    dx=[0, 1, 5, 7]
    legends=[df_out.keys()[d] for d in dx]

    # keys=df_out.keys()
    for k in dx:
        fft = tf.signal.rfft(df_out[df_out.keys()[k]])
        f_per_dataset = np.arange(0, len(fft))

        n_samples_h = len(df_out[df_out.keys()[k]])
        hours_per_year = 24*365.2524
        years_per_dataset = n_samples_h/(hours_per_year)

        f_per_year = f_per_dataset/years_per_dataset
        plt.step(f_per_year, np.abs(fft)/np.max(np.abs(fft)))
        # plt.xticks([1, 365.2524/7, 365.2524/7*2, 365.2524/7*3, 365.2524/7*4, 365.2524/7*5, 365.2524/7*6, 365.2524],
        #            labels=['1/Year','1/week', '2/week', '3/week', '4/week', '5/week', '6/week', '1/day',])
        plt.xticks([1,  365.2524/7,  365.2524,  365.2524*2],
                   labels=['1/Year','1/week',  '1/day', '2/day'], rotation=70)
        _ = plt.xlabel('Frequency (log scale)')
    plt.legend(legends)
    plt.ylabel("FFT spectrum")



def add_day_week_features(df):

    timestamp_s=[time.mktime(d.timetuple()) for d in df.index]
    timestamp_s=np.array(timestamp_s)

    day = 24*60*60
    week = 7*day
    # month=30*day
    # year = (365.2425)*day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    # df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
    # df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
    # df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    # df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    #
    return df


def data_gene(LAG_DICT, SHIFT_HEAT, df):
    keys = list(LAG_DICT.keys())
    end_date=df.index[-1]
    lag_heat_tmp=LAG_DICT['heat']
    start_date=df.index[0+SHIFT_HEAT+lag_heat_tmp]
    dates=pd.date_range(start_date,end_date,freq='1H')
    df_new=pd.DataFrame(index=dates)

    df_new['heat-lag-0']=df.loc[dates, 'heat'].to_numpy()

    for key in keys:
        print(key)

        lag_key = np.arange(LAG_DICT[key],-1,-1)
        if key=='heat':
            lag_key=lag_key+SHIFT_HEAT
        for lag in lag_key.tolist():
            start_date_lag=start_date-datetime.timedelta(hours=lag)
            end_date_lag=end_date-datetime.timedelta(hours=lag)
            dates_key=pd.date_range(start_date_lag,end_date_lag,freq='1H')
            df_new[key+'-lag-'+str(lag)] = df.loc[dates_key, key].to_numpy()

    return df_new




def add_dummy_hour(df_out):
    dummy=np.zeros(shape=(df_out.shape[0], 24))
    for k, date in enumerate(df_out.index):
        dummy[k, date.hour]=1
    names=['hour'+ str(k) for k in range(24)]
    for k, name in enumerate(names):
        df_out[name]=dummy[:,k]
    return df_out

# feature selection
def feature_selection(df1_new, X_train, y_train, alpha, n_estimators):
    """future selection using model-based methods. Two models are considered: Lasso and extraTree regression"""
    from sklearn.linear_model import Lasso
    # from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_selection import SelectFromModel
    from sklearn.feature_selection import mutual_info_regression

    keys=X_train.columns
    X_train=X_train.to_numpy()
    y_train=y_train.to_numpy()



    group=int(X_train.shape[0]/5)

    # k-fold: 5-fold for cross-validation
    feature_set_lasso=[]
    feature_set_xtree=[]
    feature_set_info=[]

    plt.figure()
    for s in range(5):
        X_val=X_train[s*group:(s+1)*group,:]
        y_val=y_train[s*group:(s+1)*group]

        plt.plot(df1_new.iloc[s*group:(s+1)*group,0])


        lgr = Lasso(alpha=alpha).fit(X_val, y_val)
        model1 = SelectFromModel(lgr, prefit=True)
        feature_set_lasso+=keys[model1.get_support()].to_list()
        print("fold:{}, selected features by LASSO: {}".format(s, keys[model1.get_support()]))


        clf = ExtraTreesRegressor(n_estimators=n_estimators)
        clf = clf.fit(X_val, y_val)
        clf.feature_importances_
        model2 = SelectFromModel(clf, prefit=True)
        feature_set_xtree+=keys[model2.get_support()].to_list()
        print("fold:{},selected features by Tree: {}".format(s, keys[model2.get_support()]))


        res= mutual_info_regression(X_val, y_val)
        res_x=res.argsort()[::-1][:5]
        feature_set_info+=keys[res_x].to_list()
        print("fold:{}, selected features by mutual information ranking: {}".format(s, keys[res_x]))

    plt.legend(["fold 1","fold 2","fold 3","fold 4","fold 5"])
    plt.ylabel("Heat load (MWh)")

    feature_set_lasso=np.unique(feature_set_lasso)
    feature_set_xtree=np.unique(feature_set_xtree)
    feature_set_info=np.unique(feature_set_info)

    print("selected features by LASSO: {}".format(feature_set_lasso))
    print("selected features by Tree: {}".format(feature_set_xtree))
    print("selected features by mutual information ranking: {}".format(feature_set_info))

    return feature_set_lasso, feature_set_xtree, feature_set_info



def data_gene_dict(LAG_DICT1, SHIFT1):
    df1_ww = pd.read_csv ('Bornholm_wea_heat.csv',sep=',', index_col=0)


    #  load files
    df1_ww.index=pd.to_datetime(df1_ww.index)
    df1_ww['windspeed']=np.sqrt(df1_ww['windx'].to_numpy()**2+df1_ww[
        'windy'].to_numpy()**2)

    df_ww_copy = df1_ww.copy()

    df_ww_copy=pd.DataFrame(columns=['heat', 'temperature', 'humidity',
                                     'DNI','windspeed'], index=df1_ww.index)
    df_ww_copy['heat']=df1_ww['heat']
    df_ww_copy['temperature']=df1_ww['temperature']
    df_ww_copy['DNI']=df1_ww['solarflux']
    df_ww_copy['windspeed']=df1_ww['windspeed']
    df_ww_copy['humidity']=df1_ww['humidity']


    df=add_day_week_features(df_ww_copy)
    print(df.index[0:10])
    df1_new=data_gene(LAG_DICT1, SHIFT1, df)

    print(df1_new.index[0:10])


    index_start=24-df1_new.index[0].hour
    index_end=1+df1_new.index[-1].hour
    df1_new=df1_new.iloc[index_start:-index_end,:]
    print(df1_new.index[0:10])
    df1_new_copy=df1_new.copy()
    # '2018-01-21 00:00:00' ~ '2020-07-05 23:00:00'


    # select the heating season data
    start0 = datetime.datetime(2019, 1, 5, 0, 0, 0);
    end0 = datetime.datetime(2019, 5, 31, 23, 0, 0);
    start1 = datetime.datetime(2019, 9, 24, 0, 0, 0);
    end1 = datetime.datetime(2020, 5, 31, 23, 0, 0);
    start2 = datetime.datetime(2020, 9, 24, 0, 0, 0);
    end2 = datetime.datetime(2021, 5, 31, 23, 0, 0);
    # start0=datetime.datetime(2018,1,22,0,0,0);
    # end0=datetime.datetime(2018,5,31,23,0,0);
    # start1=datetime.datetime(2018,9,24,0,0,0);
    # end1=datetime.datetime(2019,5,31,23,0,0);
    # start2=datetime.datetime(2019,9,24,0,0,0);
    # end2=datetime.datetime(2020,5,31,23,0,0);

    date_gene0 = pd.date_range(start=start0, end=end0, freq='H').tolist()
    date_gene1 = pd.date_range(start=start1, end=end1, freq='H').tolist()
    date_gene2 = pd.date_range(start=start2, end=end2, freq='H').tolist()

    dates = date_gene0 + date_gene1 + date_gene2


    # 3:1 for train and test
    df1_new=df1_new.loc[dates,:]

    return df1_new

def rmse_func( df_x):
    columns = df_x.columns
    col_base = columns[0]

    df_metric = pd.DataFrame(index=columns[1:], columns=['RMSE'])

    for col in columns[1:]:
        df_metric.loc[col] = np.linalg.norm(df_x[col_base].to_numpy() - df_x[col].to_numpy()) * np.sqrt(
            1 / df_x.shape[0])

    return df_metric
def mae_func( df_x):
    columns = df_x.columns
    col_base = columns[0]

    df_metric = pd.DataFrame(index=columns[1:], columns=['MAE'])

    for col in columns[1:]:
        df_metric.loc[col] = np.abs(df_x[col_base].to_numpy() - df_x[col].to_numpy()).mean()

    return df_metric
def r2_func( df_x):
    columns = df_x.columns
    col_base = columns[0]

    df_metric = pd.DataFrame(index=columns[1:], columns=['R2'])

    for col in columns[1:]:
        df_metric.loc[col] = 1 - np.sum((df_x[col_base].to_numpy() - df_x[col].to_numpy()) ** 2) / np.sum((df_x[
                                                                                                     col_base].to_numpy() -
                                                                                                 df_x[
                                                                                                     col_base].to_numpy().mean()) ** 2)

    return df_metric
def bias_func( df_x):
    columns = df_x.columns
    col_base = columns[0]

    df_metric = pd.DataFrame(index=columns[1:], columns=['Bias'])

    for col in columns[1:]:
        df_metric.loc[col] = df_x[col_base].to_numpy().mean() - df_x[col].to_numpy().mean()
    return df_metric

def rmse_array_func( df_x):
    columns = df_x.columns
    col_base = columns[0]

    df_metric = pd.DataFrame(index=np.arange(24), columns=columns[1:])

    for col in columns[1:]:
        df_metric[col] = np.sqrt(((df_x[col_base].to_numpy().reshape(-1,24) - df_x[col].to_numpy().reshape(-1,
                                                                                                          24))**2).mean(axis=0))

    return df_metric

def irmse_func(df_rmse):
    """df_xrmse is dataframe with index of methods and column of RMSE """
    rmse_bench=df_rmse.loc["Persistent","RMSE"] # a scalar
    # df_rmse1=df_rmse.drop('Persistent')
    df_rmse1=df_rmse.copy()
    df_irmse=pd.DataFrame(index=df_rmse1.index, columns=['IRMSE'])
    for idx in df_irmse.index:
        df_irmse.loc[idx,'IRMSE']=(rmse_bench-df_rmse1.loc[idx,'RMSE'])/rmse_bench*100
    return df_irmse

def imae_func(df_rmse):
    """df_xrmse is dataframe with index of methods and column of RMSE """
    rmse_bench=df_rmse.loc["Persistent","MAE"] # a scalar
    # df_rmse1=df_rmse.drop('Persistent')
    df_rmse1=df_rmse.copy()
    df_irmse=pd.DataFrame(index=df_rmse1.index, columns=['IMAE'])
    for idx in df_irmse.index:
        df_irmse.loc[idx,'IMAE']=(rmse_bench-df_rmse1.loc[idx,'MAE'])/rmse_bench*100
    return df_irmse


def irmse_array_func(df_rmse):
    """df_xrmse is dataframe with index of methods and column of RMSE """
    rmse_bench=df_rmse["Persistent"].to_numpy() # a scalar
    df_rmse1=df_rmse.copy()
    # df_rmse1.pop('Persistent')
    df_irmse=pd.DataFrame(index=df_rmse1.index, columns=df_rmse1.columns)
    for col in df_irmse.columns:
        df_irmse[col]=(rmse_bench-df_rmse1[col].to_numpy())/rmse_bench*100
    return df_irmse

def mae_array_func( df_x):
    columns = df_x.columns
    col_base = columns[0]

    df_metric = pd.DataFrame(index=np.arange(24), columns=columns[1:])

    for col in columns[1:]:
        df_metric[col] = np.abs(df_x[col_base].to_numpy().reshape(-1,24) - df_x[col].to_numpy().reshape(-1,24)).mean(axis=0)

    return df_metric
def r2_array_func( df_x):
    columns = df_x.columns
    col_base = columns[0]

    df_metric = pd.DataFrame(index=np.arange(24), columns=columns[1:])

    for col in columns[1:]:
        df_metric[col] = 1 - np.sum((df_x[col_base].to_numpy().reshape(-1,24) - df_x[col].to_numpy().reshape(-1,
                                                                                                                 24))
                                        ** 2, axis=0) / np.sum((df_x[col_base].to_numpy().reshape(-1,24) -df_x[
                                                                                                     col_base].to_numpy().reshape(-1,24).mean(axis=0)) ** 2, axis=0)

    return df_metric


def bias_array_func( df_x):
    columns = df_x.columns
    col_base = columns[0]

    df_metric = pd.DataFrame(index=np.arange(24), columns=columns[1:])

    for col in columns[1:]:
        df_metric[col] = df_x[col_base].to_numpy().reshape(-1,24).mean(axis=0) - df_x[col].to_numpy().reshape(-1,
                                                                                                             24).mean(axis=0)
    return df_metric



def irmse_func(df_rmse):
    """df_xrmse is dataframe with index of methods and column of RMSE """
    rmse_bench=df_rmse.loc["Persistent", "RMSE"] # a scalar
    # df_rmse1=df_rmse.drop('Persistent')
    df_rmse1=df_rmse.copy()
    df_irmse=pd.DataFrame(index=df_rmse1.index, columns=['IRMSE'])
    for idx in df_irmse.index:
        df_irmse.loc[idx,'IRMSE']=(rmse_bench-df_rmse1.loc[idx,'RMSE'])/rmse_bench*100
    return df_irmse

def imae_func(df_rmse):
    """df_xrmse is dataframe with index of methods and column of RMSE """
    rmse_bench=df_rmse.loc["Persistent","MAE"] # a scalar
    # df_rmse1=df_rmse.drop('Persistent')
    df_rmse1=df_rmse.copy()
    df_irmse=pd.DataFrame(index=df_rmse1.index, columns=['IMAE'])
    for idx in df_irmse.index:
        df_irmse.loc[idx,'IMAE']=(rmse_bench-df_rmse1.loc[idx,'MAE'])/rmse_bench*100
    return df_irmse




def imae_array_func(df_rmse):
    """df_xrmse is dataframe with index of methods and column of RMSE """
    rmse_bench=df_rmse["Persistent"].to_numpy() # a scalar
    df_rmse1=df_rmse.copy()
    # df_rmse1.pop('Persistent')
    df_irmse=pd.DataFrame(index=df_rmse1.index, columns=df_rmse1.columns)
    for col in df_irmse.columns:
        df_irmse[col]=(rmse_bench-df_rmse1[col].to_numpy())/rmse_bench*100
    return df_irmse


