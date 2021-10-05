__author__ = 'Li Bai'

# Data introduction
import pandas as pd
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import multiprocessing
# from functools import reduce

season_heat = [1, 2, 3, 4, 5, 9, 10, 11, 12]
season_heat_non = [6, 7, 8]

def addrs_list(path, path1, path2):
    addrs=[]
    addrsx=[]
    meters=[]
    addrs_select=[]
    # for file in os.listdir("./Bornholmdata/energy/"):
    #     if file.endswith(".csv"):
    #         addrs.append(os.path.join("./Bornholmdata/energy/", file))
    for file in os.listdir(path):
        if file.endswith(".csv"):
            meters.append(file[0:-4])
            addrs.append(os.path.join(path, file))
            addrsx.append(os.path.join(path1, file))
            addrs_select.append(os.path.join(path2, file))
            # addrs_pkl.append(os.path.join(path2, file[0:-4]+".pkl"))

    return addrs, addrsx, addrs_select, meters


def filler_null_meter(df_meter, season):
    """df_meter is dataframe with the index column of time stamps and a column of energy:
    2 strategies are adopted to fill the null values: the first one is to fill it with the average value of the
    current year and current month; if the average contains at least one null value, then the null values will be
    replaced with the total average """

    df_meter1=df_meter.copy()
    df_meter2=pd.DataFrame(index=df_meter.index, columns=df_meter1.columns.tolist()+['mean_whole','mean_current_month'])
    df_meter2['heat']=df_meter1['heat']
    df_meter1['hour']=df_meter1.index.hour
    df_meter1_mean=df_meter1.groupby('hour').mean()
    for hour in range(24):
        index=df_meter1.loc[df_meter1['hour']==hour].index
        df_meter2.loc[index, 'mean_whole']=df_meter1_mean.iloc[hour,0]
    df_meter1['month']=df_meter1.index.month
    df_meter1['year']=df_meter1.index.year
    flag=0

    for year in [2019,2020,2021]:
        for month in season:
            if (year==df_meter1.index[-1].year)&(month==df_meter1.index[-1].month):
                flag=1

            df_meter1_month=df_meter1.loc[(df_meter1['month']==month)&(df_meter1['year']==year)].copy()
            df_meter1_month.pop('month')
            df_meter1_month.pop('year')
            df_meter1_month_mean = df_meter1_month.groupby('hour').mean()

            for hour in range(24):
                index = df_meter1_month.loc[df_meter1_month['hour'] == hour].index
                # print(index)
                # print(year);
                # print(month);
                # print(hour)
                df_meter2.loc[index, 'mean_current_month'] = df_meter1_month_mean.iloc[hour, 0]

            if flag == 1:
                break
    df_meter3=df_meter2.copy()
    index1=df_meter3.loc[df_meter3['heat'].isnull(), 'heat'].index
    df_meter3.loc[index1, 'heat']= df_meter3.loc[index1, 'mean_current_month'].to_numpy()

    index2=df_meter3.loc[df_meter3['heat'].isnull(), 'heat'].index
    df_meter3.loc[index2, 'heat']= df_meter3.loc[index2, 'mean_whole'].to_numpy()

    df_meter3.pop('mean_current_month')
    df_meter3.pop('mean_whole')
    return df_meter3

def filler_null_meter_complete(addrx, addr_select,  gap_max, gap_sum):

    # ensure that addrx, gap_max, gap_sum are from the same meter
    df_meter=pd.read_csv(addrx, index_col='Unnamed: 0')
    df_meter=df_meter.set_index('timestamp')
    df_meter.index=pd.to_datetime(df_meter.index)

    if (gap_max<=48)&(gap_sum<=24*90):
        df_meter1 = df_meter.copy()
        df_meter1['hour'] = df_meter1.index.hour
        df_meter1['month'] = df_meter1.index.month

        df_meter1_heat = df_meter1.loc[
                          ((df_meter1.month >= 1) & (df_meter1.month <= 5)) | ((df_meter1.month >= 9) &
                                                                                 (df_meter1.month <= 12)), :]

        df_meter1_heat.pop('hour')
        df_meter1_heat.pop('month')

        df_meter2_heat=filler_null_meter(df_meter1_heat, season_heat)

        df_meter1_heat_non = df_meter1[((df_meter1.month >= 6) & (df_meter1.month <= 8))]

        df_meter1_heat_non.pop('hour')
        df_meter1_heat_non.pop('month')

        df_meter2_heat_non = filler_null_meter(df_meter1_heat_non, season_heat_non)

        df_meter2=pd.concat([df_meter2_heat, df_meter2_heat_non], axis=0)
        df_meter2=df_meter2.sort_index()

        df_meter2.to_csv(addr_select)





#
#
#
#
#


if __name__ == '__main__':

    path='./meter/'
    import os

    path1 = './meter_continous/'
    if not os.path.exists(path1):
        os.makedirs(path1)

    path2= './meter_select/'
    if not os.path.exists(path2):
        os.makedirs(path2)


    _, addrsx, addrs_select, meters=addrs_list(path, path1, path2)
    df_gap=pd.read_csv("gaps.csv")
    df_gap_meters=pd.DataFrame(columns=['meter'])
    meters_int=[int(meter) for meter in meters]
    df_gap_meters['meter']=meters_int
    df_gap_merge=pd.merge(df_gap_meters, df_gap, how='left', on='meter')

    meters=df_gap_merge['meter'].tolist()
    gap_sums=df_gap_merge['gap_sum'].tolist()
    gap_maxs=df_gap_merge['gap_max'].tolist()


    with multiprocessing.Pool(processes=24) as pool:
        pool.starmap(filler_null_meter_complete, zip(addrsx, addrs_select, gap_maxs, gap_sums))
#