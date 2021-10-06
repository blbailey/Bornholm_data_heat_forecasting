__author__ = 'Li Bai'
# Data introduction

"""aggregated some meters if they are from the same zone based on the way of numbering the meters"""

import os
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
mpl.rc('xtick', labelsize=20)
mpl.rc('ytick', labelsize=20)
plt.rcParams.update({'font.size': 24})

mpl.rcParams['figure.figsize'] = (16,10)
mpl.rcParams['axes.grid'] = False


# aggregate the heat data
path2='./meter_select/'
path3='./meter_reselect/'

# ============select each zone based on the first several digits in the file name===========
if not os.path.exists(path3):
    os.makedirs(path3)

def addrs_list(path2, path3):
    addrs=[]
    addrs_resel=[]
    file_digits=[]
    for file in os.listdir(path2):
        if file.endswith(".csv"):
            addrs.append(os.path.join(path2, file))
            addrs_resel.append(os.path.join(path3, file))
            file_digits.append(int(file[0:3]))
    return addrs, file_digits, addrs_resel

addrs, digits, addrs_resel=addrs_list(path2, path3)


dy = pd.read_csv(addrs[0], index_col='timestamp');
dy.index = pd.to_datetime(dy.index)

df_sum = pd.DataFrame(columns=['heat'], index=dy.index)
sumx = np.zeros(shape=dy.shape)
null_nums=[]
addrs_new=[]
for k, addr in enumerate(addrs):
    dy = pd.read_csv(addr, index_col='timestamp');
    dy.index = pd.to_datetime(dy.index)

    null_num=np.isnan(dy.to_numpy()).sum()
    addr_resel=addrs_resel[k]
    if null_num==0:
        addrs_new.append(addr)
        sumx = sumx + dy.to_numpy()
        # dy.to_csv(addr_resel)
    null_nums.append(null_num)
df_sum['heat']= sumx





# D:\OneDrive\OneDrive - Danmarks Tekniske Universitet\energydataDTU\venv\bornholm2021\Bornholm_hourly_forecast_online_hyperpara.py
DIR1="Bornholm_weather00_full_2021.csv"

wea=pd.read_csv(DIR1)
wea['time']=pd.to_datetime(wea['time'])
df_merge=pd.merge(df_sum, wea, how='left', left_on='timestamp',right_on='time')
df_merge=df_merge.iloc[24:,:]
print(df_merge.columns)
df_merge=df_merge.set_index('time')
df_merge.to_csv("Bornholm_wea_heat.csv")



