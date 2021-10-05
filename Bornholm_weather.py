__author__ = 'Li Bai'
"""data description: whether data are fetched directly from Norway Meteo 
website; the forecast data are updated every 6 hours a day for the next 48 
hours; the parameters are variables

1)	'air_temperature_2m',
2)	'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time',
3)	'precipitation_amount_acc',
4)	'relative_humidity_2m',
5)	'x_wind_10m',
6)	'y_wind_10m'

All the data will be processed in a way that they finally will be put into a 
file of 12x31x24=8928 samples

reorganize the data in the fetched files; remove null data and fill with yesterday's data
"""



import pandas as pd
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt

year=['2019','2020','2021']
month=["01","02","03","04","05","06","07","08","09","10","11","12"]
path_root="Bornholm_weather"
path_root_new='Bweather'

def path_change(path_root, year, month):
    addrs = []
    newpath = path_root + "/" + year + month
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for file in os.listdir(path_root):
            if file.endswith(year+month+".csv"):
                os.rename(path_root+"/"+file,
                          newpath+"/"+file)

                addrs.append(newpath+"/"+file)
    return addrs

def file_move(path_root, year, month):
    for ky in range(len(year)):
        for km in range(len(month)):
            path_change(path_root, year=year[ky], month=month[km])


# addrs=path_read(path_root)
# addrs_month=addrs[0]
# 50 columns: first column is the updated time and the others are for the 49
# hours including this one; 49 hours are selected because of accumulated data
# we have to convert them for single hours
#  every one should select updated hour[0,1,2,3] refers to 00 06 12 18 and
#  variables [0,1,2,3,4,5]
def file_month(addrs_month,  index_var, hour_update, names):

    var1 = pd.read_csv(addrs_month[index_var], sep=",")
    col0 = var1.columns[0]
    var1_new = var1.rename(columns={col0: 'time'})

    var1_new['time'] = [datetime.datetime.utcfromtimestamp(t) for t in var1_new[
        'time']]
    var1_new= var1_new.set_index('time')

    if names[index_var] in ['solarflux','precipitation']:


        cols_0_48=var1_new.iloc[:,0:-1].to_numpy()
        cols_1_49=var1_new.iloc[:,1:].to_numpy()


        var1_new_1=pd.DataFrame(columns=var1_new.columns[:-1],
                                index=var1_new.index)

        var1_new_1.iloc[:,0:]=cols_1_49-cols_0_48

        var1_new_x=var1_new_1.iloc[hour_update::4,:]
    else:
        var1_new_x=var1_new.iloc[hour_update::4,:-1]
    # with [:, idx:] here idx refers to the values of tomorrow.
    # update_hour=0, idx=24; update=6; idx=18:18+24 '
    # update_hour=12, idx=12:12+24; update_hour=24; idx=0:24
    # idx=24-update_hour: 24-update_hour+24
    return var1_new_x.iloc[:,(24-hour_update): (24-hour_update+24)]

def file_all(addrs, index_var, hour_update, names):
    dfs=[]
    for addrs_month in addrs:
        df=file_month(addrs_month,  index_var, hour_update, names)
        dfs.append(df)
    dfs=pd.concat(dfs)
    return dfs
# =======================================================================

# generate addrs
def addrs_gene():
    # names=['temperature','solarflux','precipitation','humidity','windx','windy']
    names=['integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time',
     'air_temperature_2m', 'relative_humidity_2m',
     'precipitation_amount_acc', 'x_wind_10m',
     'y_wind_10m']
    years=['2019','2020','2021']
    months=['01', '02', '03','04', '05', '06','07', '08', '09','10', '11', '12']
    addrs=[]
    for year in years:
        for month in months:
            addrs_month=[]
            for name in names:
                addr='Bornholm_weather/Bornholm_'+name+'_'+year+month+'.csv'
                addrs_month.append(addr)
            addrs.append(addrs_month)
    return addrs



def subtract_update_data(path_root_new, names):
    addrs = addrs_gene()
    # names=['solarflux',
    #  'temperature', 'humidity',
    #  'precipitation', 'windx',
    #  'windy']

    # names=['temperature','solarflux','precipitation','humidity','windx','windy']
    hours=['00','06','12','18']
    import os
    if not os.path.exists(path_root_new):
        os.makedirs(path_root_new)
    # every one should select updated hour[0,1,2,3] refers to 00 06 12 18 and
    #  variables [0,1,2,3,4,5] refers to names
    for hour_update in range(len(hours)):
        for index_var in range(6):
            dfs=file_all(addrs, index_var=index_var, hour_update=hour_update, names=names)
            dfs.to_csv('Bweather/'+names[index_var] +'_'+hours[
                hour_update]+'.csv',
                       sep=',')
            # dfs.plot()


#
#
def path_gene(path_root_new, names):
    addrs = []
    for name in names:
        # if file.endswith("00.csv"):
        addrs.append(path_root_new + "/" + name+'_00.csv')
    return addrs
# ['./Bornholm_weather/humidity_00.csv',
#  './Bornholm_weather/precipitation_00.csv',
#  './Bornholm_weather/solarflux_00.csv',
#  './Bornholm_weather/temperature_00.csv',
#  './Bornholm_weather/windx_00.csv',
#  './Bornholm_weather/windy_00.csv']
# =================================================================================

def data_reshape_column(addrs_new, names):
    dfs=[]
    for namek in range(len(addrs_new)):
        var = pd.read_csv(addrs_new[namek], sep=",", index_col='time')
        var.index=pd.to_datetime(var.index)
        start_date=var.index[0]+datetime.timedelta(hours=24)
        end_date=var.index[-1]+datetime.timedelta(hours=47)
        dates=pd.date_range(start_date,end_date,freq='H')
        col_name=names[namek]
        df_var=pd.DataFrame(columns=['time',col_name])
        df_var['time']=dates
        df_var=df_var.set_index('time')
        print((var.iloc[:,0:].to_numpy()).shape)
        if col_name=='temperature':
            df_var[col_name]=((var.iloc[:,0:].to_numpy()).reshape(-1,1))-273.15
        else:
            df_var[col_name] = (var.iloc[:, 0:].to_numpy()).reshape(-1, 1)
        dfs.append(df_var)
    dfs=pd.concat(dfs,axis=1)
    return dfs



# ==============read data from each files in the directory of Nordhavn_weather and then put it into the Bweather1 as 00

# this is to read files from Nordhavn_weather and to save it to Bweather1/Bornholm_weather00.csv

# followng alphabetic order
names = ['solarflux',
         'temperature', 'humidity',
         'precipitation', 'windx',
         'windy']
subtract_update_data(path_root_new, names)

addrs_new=path_gene(path_root_new, names)
dfs=data_reshape_column(addrs_new, names)
# dfs.to_csv('./Nweather/Nordhavn_weather_total.csv',
#                        sep=',')


def fill_nulls(dfs):
    dfs[dfs==-99.99]=np.nan
    dfs.loc[dfs['temperature']==-273.15-99.99,'temperature']=np.nan

    # all the data are missing for the same date; so we do it for all columns
    # together

    cols=dfs.columns
    for col in cols:
        dfs_null=dfs.loc[dfs[col].isna()==True]
        for idx in dfs_null.index:
            idx_yes=idx-datetime.timedelta(hours=24)
            dfs.loc[idx, col]=dfs.loc[idx_yes, col]
    return dfs


dfs=fill_nulls(dfs)
dfs.to_csv("Bornholm_weather00_full_2021"
           ".csv")



# # print(dfs.iloc[0,:]
# Out[8]:
# humidity           0.568547
# precipitation     -0.001953
# solarflux          0.000000
# temperature      277.342285
# windx              5.727576
# windy            -14.121002
# Name: 2019-01-01 00:00:00, dtype: float64)





















