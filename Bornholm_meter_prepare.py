__author__ = 'Li Bai'
# Data introduction
import pandas as pd
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
# import pickle
import multiprocessing

# from functools import reduce





def addrs_list(path, path1):
    addrs=[]
    addrsx=[]
    addrs_pkl=[]
    meters=[]
    # for file in os.listdir("./Bornholmdata/energy/"):
    #     if file.endswith(".csv"):
    #         addrs.append(os.path.join("./Bornholmdata/energy/", file))
    for file in os.listdir(path):
        if file.endswith(".csv"):
            meters.append(file[0:-4])
            addrs.append(os.path.join(path, file))
            addrsx.append(os.path.join(path1, file))
            # addrs_pkl.append(os.path.join(path2, file[0:-4]+".pkl"))

    return addrs, addrsx, meters


def check_time_continous(addr, addrx, meter):
    df_meter = pd.read_csv(addr)
    df_meter['timestamp']=pd.to_datetime(df_meter['timestamp'])

    # define the same dates
    start_time='2019-01-01 01:00:00'#df_meter['timestamp'].iloc[0]
    end_time='2021-08-05 02:00:00'#df_meter['timestamp'].iloc[-1]
    dates=pd.date_range(start_time, end_time, freq='H')

    # if the data timestamp are the same, merge them!
    df_meter = df_meter.groupby(by='timestamp').mean()
    df_meter=df_meter.reset_index()

    df_meter_new=pd.DataFrame(columns=['timestamp'])
    df_meter_new['timestamp']=dates
    df_meters=pd.merge(df_meter_new, df_meter, how='left', on='timestamp')

    # outliers occasion 1: if a single value is higher than 0.1, directly put it to be np.nan
    df_meters.loc[df_meters.heat>=0.1, 'heat']=np.nan
    df_meters.loc[df_meters.heat<=0., 'heat']=0.

    # heating season [10,11,12,1,2,3,4,5] and non-heating season [6,7,8] the rest!:
    df_meters1=df_meters.copy()
    df_meters1['hour']=df_meters1['timestamp'].dt.hour
    df_meters1['month']=df_meters1['timestamp'].dt.month

    df_meters1_heat=df_meters1.loc[((df_meters1.month>=1) & (df_meters1.month<=5)) | ((df_meters1.month>=9) &
                                                                               (df_meters1.month<=12)),:]

    df_meters2_heat=df_meters1_heat.copy()
    df_meters1_heat.pop('timestamp')
    df_meters1_heat.pop('month')

    df_meters1_heat_quantile25=df_meters1_heat.groupby('hour').quantile(0.25)
    df_meters1_heat_quantile75=df_meters1_heat.groupby('hour').quantile(0.75)

    df_meters1_heat_upper=df_meters1_heat_quantile75+1.5*(df_meters1_heat_quantile75-df_meters1_heat_quantile25)
    df_meters1_heat_lower=df_meters1_heat_quantile25 # all the values are zero...

    for hour in range(24):
        df_tmp_heat_hour=df_meters2_heat.loc[(df_meters2_heat.hour==hour)]
        df_tmp_heat_hour1=df_tmp_heat_hour.loc[(df_tmp_heat_hour.heat)>= df_meters1_heat_upper.iloc[hour,0]]
        index=df_tmp_heat_hour1.index
        df_meters2_heat.loc[index, 'heat']=np.nan

    df_meters2_heat.pop('hour')
    df_meters2_heat.pop('month')






    df_meters1_heat_non=df_meters1[((df_meters1.month>=6) & (df_meters1.month<=8))]
    df_meters2_heat_non=df_meters1_heat_non.copy()
    df_meters1_heat_non.pop('timestamp')
    df_meters1_heat_non.pop('month')

    df_meters1_heat_non_quantile25=df_meters1_heat_non.groupby('hour').quantile(0.25)
    df_meters1_heat_non_quantile995=df_meters1_heat_non.groupby('hour').quantile(0.995)

    df_meters1_heat_non_upper=df_meters1_heat_non_quantile995#+1.5*(
    # df_meters1_heat_non_quantile75-df_meters1_heat_non_quantile25)
    df_meters1_heat_non_lower=df_meters1_heat_non_quantile25 # all the values are zero...

    for hour in range(24):
        df_tmp_heat_non_hour=df_meters2_heat_non.loc[(df_meters2_heat_non.hour==hour)]
        df_tmp_heat_non_hour1=df_tmp_heat_non_hour.loc[(df_tmp_heat_non_hour.heat)>= df_meters1_heat_non_upper.iloc[hour,0]]
        index=df_tmp_heat_non_hour1.index
        df_meters2_heat_non.loc[index, 'heat']=np.nan

    df_meters2_heat_non.pop('hour')
    df_meters2_heat_non.pop('month')

    df_meters2=pd.concat([df_meters2_heat, df_meters2_heat_non],axis=0, join='outer')
    df_meters2.to_csv(addrx)
    gap_max, gap_sum=count_continuous_null(df_meters2)
    return gap_max, gap_sum, meter


# # if the meter has nonzeros values higher than 0.1, put it to nan
# # fix the date to be from 2019-Jan-01, 00:00 to 2020-June-30 23:00
def count_continuous_null(x_in):
    # print("current file is {}".format(file))
    # x=pd.read_csv(file)
    # x=x.set_index("time")
    # x.index=pd.to_datetime(x.index)
    sts=[]
    end=[]
    x=x_in.copy()
    x=x.set_index('timestamp')
    times=x.index
    if x.loc[times[0]].isnull().sum() == 1:
        sts.append(0)
    for k in range(len(times)-1):
        if (x.loc[times[k]].isnull().sum()==1)&(x.loc[times[k+1]].isnull(
            ).sum()==0):
            end.append(k+1)
        if (x.loc[times[k]].isnull().sum()==0)&(x.loc[times[k+1]].isnull(
            ).sum()==1):
            sts.append(k+1)
    if x.loc[times[-1]].isnull().sum()==1:
        end.append(len(times))
    gaps=np.array(end)-np.array(sts)
    if gaps.shape[0]==0:
        gap_max=0
        gap_sum=0
    else:
        gap_max=gaps.max()
        gap_sum=gaps.sum()

    # dict_file={"loc_start": sts, "gaps":gaps.tolist(), "gap_max": gap_max,
    #            "gap_sum": gap_sum}
    #
    #
    #
    # f=open(addr_pkl,'wb')
    # pickle.dump(dict_file, f)
    # f.close()

    return gap_max, gap_sum


if __name__ == '__main__':

    path = './meter/'


    path1 = './meter_continous/'
    if not os.path.exists(path1):
        os.makedirs(path1)


    addrs, addrsx, meters=addrs_list(path, path1)


    with multiprocessing.Pool(processes=24) as pool:
        results=pool.starmap(check_time_continous, zip(addrs, addrsx, meters))

    df_gap=pd.DataFrame(columns=['gap_max','gap_sum','meter'])
    for k in range(len(results)):
        df_gap.loc[k]=list(results[k])
    df_gap.to_csv("gaps.csv", index_label=False)


