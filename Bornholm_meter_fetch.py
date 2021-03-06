__author__ = 'Li Bai'


# This file is to download the data of twp apartments from API


import requests
import json
import numpy as np
import datetime as datetime
import pandas as pd
import csv

# =========================explanation of the data on energydata.dk=========================

# ================================================================================================


headers = {
    'accept': 'application/json',
    'authorization':
    'Bearer '
    # 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjVhMGFkNTgzNTRmYjUxMWFkYjUxNzcyM2RlYmI3Yzc5YmZhMTY4OGVjMDg5ZjY5NTAxMzhlZGJhMWY1ODJmNTU3MmE4ZDhmZDQ3Y2FmMmExIn0.eyJhdWQiOiIxIiwianRpIjoiNWEwYWQ1ODM1NGZiNTExYWRiNTE3NzIzZGViYjdjNzliZmExNjg4ZWMwODlmNjk1MDEzOGVkYmExZjU4MmY1NTcyYThkOGZkNDdjYWYyYTEiLCJpYXQiOjE1OTg2MjA3NDIsIm5iZiI6MTU5ODYyMDc0MiwiZXhwIjoxNjMwMTU2NzQyLCJzdWIiOiIxMTMiLCJzY29wZXMiOltdfQ.Nzv6R6_ypIMPrbIMwOT92aXxtgY8SfkRKsZsKu6nOjq2fx75g-DpSC4fk-CGh30eP5krFwdzqYqBrS4tRu-fS3109Kf5otjf9Kh4jujGO9mK4pXph1ON5ihp00wM2Naiyjn4Le07VLM5TpOgbpkVp3q_f6iCOv_JJjydB30THdvZW1k_ikt-oHlLfiHKeVuJdxMQTZMPbK9Sy5QAeznDlZkKXR51-8fvD1mWGnxs2vJh7MgrIFLAx2Xb5QzttQGZx4pXOcqX3yLvC0qqnDdd_1AdZ-Te0X9Wb8mefufM8b8N4Mi8-rpJwL6b9YgYc5KMBLGakD_ZJynvHU2NOIFrdFDD_WBZaDvtgHUSx72y3DqX2E2MDfpZFacV09OLlu_0JZ_NDl6wcbBGRgN3XqLwzWzGCI2ZU5jkUANuRx2a1wkWk9Vzg0jFTv_rDl8edLn2FPzgsvuHyO5atyxVqACqXQRbKj2tFHNKjQKYYdeG8mDhM0mY1I0mpUvIEMyeVGAf6cyEETmwXtQBW9Id_cjulXDGZBegYJLP--elBfWMcWADM8AO99CVzh0A5_gg1_wHFvCxb0HRWaU3_BV0RR3YhFxU-ogCNdY-9NZ2injUSOzGK4E-gdiaoSQKVMghDcqXXNaHZeOkz4KY1DgpyqJKEqBzuPDccu1obn3yUYvaqeA',    # 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjVhMGFkNTgzNTRmYjUxMWFkYjUxNzcyM2RlYmI3Yzc5YmZhMTY4OGVjMDg5ZjY5NTAxMzhlZGJhMWY1ODJmNTU3MmE4ZDhmZDQ3Y2FmMmExIn0.eyJhdWQiOiIxIiwianRpIjoiNWEwYWQ1ODM1NGZiNTExYWRiNTE3NzIzZGViYjdjNzliZmExNjg4ZWMwODlmNjk1MDEzOGVkYmExZjU4MmY1NTcyYThkOGZkNDdjYWYyYTEiLCJpYXQiOjE1OTg2MjA3NDIsIm5iZiI6MTU5ODYyMDc0MiwiZXhwIjoxNjMwMTU2NzQyLCJzdWIiOiIxMTMiLCJzY29wZXMiOltdfQ.Nzv6R6_ypIMPrbIMwOT92aXxtgY8SfkRKsZsKu6nOjq2fx75g-DpSC4fk-CGh30eP5krFwdzqYqBrS4tRu-fS3109Kf5otjf9Kh4jujGO9mK4pXph1ON5ihp00wM2Naiyjn4Le07VLM5TpOgbpkVp3q_f6iCOv_JJjydB30THdvZW1k_ikt-oHlLfiHKeVuJdxMQTZMPbK9Sy5QAeznDlZkKXR51-8fvD1mWGnxs2vJh7MgrIFLAx2Xb5QzttQGZx4pXOcqX3yLvC0qqnDdd_1AdZ-Te0X9Wb8mefufM8b8N4Mi8-rpJwL6b9YgYc5KMBLGakD_ZJynvHU2NOIFrdFDD_WBZaDvtgHUSx72y3DqX2E2MDfpZFacV09OLlu_0JZ_NDl6wcbBGRgN3XqLwzWzGCI2ZU5jkUANuRx2a1wkWk9Vzg0jFTv_rDl8edLn2FPzgsvuHyO5atyxVqACqXQRbKj2tFHNKjQKYYdeG8mDhM0mY1I0mpUvIEMyeVGAf6cyEETmwXtQBW9Id_cjulXDGZBegYJLP--elBfWMcWADM8AO99CVzh0A5_gg1_wHFvCxb0HRWaU3_BV0RR3YhFxU-ogCNdY-9NZ2injUSOzGK4E-gdiaoSQKVMghDcqXXNaHZeOkz4KY1DgpyqJKEqBzuPDccu1obn3yUYvaqeA',
    'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6ImM2Mjc4ZTYxZmI0MmY1MDg4NWNjNTg2NWNmYWU1NTkzNWY3MjgzMDZlYTQ4MmJlOTEyODI3MWYzN2Q1NTQ3YjA1NjMyZGFhYmE2NjM0ZWEyIn0.eyJhdWQiOiIxIiwianRpIjoiYzYyNzhlNjFmYjQyZjUwODg1Y2M1ODY1Y2ZhZTU1OTM1ZjcyODMwNmVhNDgyYmU5MTI4MjcxZjM3ZDU1NDdiMDU2MzJkYWFiYTY2MzRlYTIiLCJpYXQiOjE2MjgxNDc1ODksIm5iZiI6MTYyODE0NzU4OSwiZXhwIjoxNjU5NjgzNTg5LCJzdWIiOiIxMjQiLCJzY29wZXMiOltdfQ.RzRuzavNYEU0jtMBzbbtZawClJ16Iv1W1TNhMiEdqI4tUFdoUkqSrGRNSHfLe2xMdFN28alP58SbZ3C3CuChIFNOkSrAy6KC_hwohmEu1PUr-gJ1M7mnhwdAHSkA5rCtvXTsw9eM5QLHYN82EMpUvVkUPaK6S7m0DczH_6db1nKWYQmh7cU7yn-Onh78kkSmxYvq83njtdlLOphM4YQaVxz-7DvIkunPt-UiXs__e-JGDGCSFbIq2_p1OtiPo2atqYyheucHLVa1CEzfXjkn9xRUfZlD3Bfd2XwQ4KM6Jhx8e4w8uOhPCDLg0tAyLQxpltZMWLDfz3SsB9LuVcGI9A-psbEAUmegbsj-pVAmFwp1M8pid2NJ7NJhPv5KK-gdTngiEIzpJWQv-leJHxmP3_hWAliOEOVLhZjW1_PniXVy9zOmZlmarzjoa7XOuVcxCGaO-o5Fa6MEgAJe29dtOTFG3VGWUBlMoEGEFqKb3Joa3snX8zXBv-RsoTRWQi5QktDmirXWGSSrlH1xlbhgKZZuyqxV-PrV_sv5LzkJBIW505CPwiOHrcj6AJoGnAwV9Ky5uDkZGiD6cbRYYZmImc0mpi4__5wTCphM-ArFMBPh0gnKUF4jcc2XpEDVyZlz7xIxg7vuWcuL2bQDPf-uGvagII5h4Z7vbWKYfSup6v8',
}

time_from ='2019-01-01T00:00:00'
time_to='2022-12-31T23:00:00'


#  property_ids

meters_hourly=pd.read_csv("BEOF_hourly_meter_list.csv", header=None)
ids=pd.read_csv("bornholms-varme-fast-properties.csv")
topic_energy_list=['9F7P/00/00/BornholmsVarme/energy/'+str(meter) for meter in meters_hourly.iloc[:,0]]
df_topic=pd.DataFrame(columns=['topic'])
df_topic['topic']=topic_energy_list
df_energy=pd.merge(df_topic, ids, how='inner', on='topic')
ids=df_energy['property_id'].tolist()
meters=meters_hourly.iloc[:,0].tolist()
for k, id in enumerate(ids):

    r = requests.get('https://api.energydata.dk/api/properties/values?ids=' + str(id)
                     + '&from=' + time_from + '&to=' + time_to, headers=headers)
    r_content=r.content.decode('utf-8')
    cr=csv.reader(r_content.splitlines(),delimiter=',')
    my_list=list(cr)
    arr_cr=np.array(my_list)
    arr_cr=arr_cr.astype(np.float)
    # the column 0 is the property id, column 1 is the timestamp, and column 2 is the real value
    timestamps=[datetime.datetime.fromtimestamp(int(t)/1000).strftime("%Y-%m-%d %H:%M:%S") for t in arr_cr[:,1]]
    df_cr=pd.DataFrame(columns=['timestamp','heat'])
    df_cr['timestamp']=timestamps;
    df_cr=df_cr.set_index('timestamp')
    df_cr['heat']=arr_cr[:,2]

    import os
    path="meter/"
    if not os.path.exists(path):
        os.makedirs(path)
    df_cr.to_csv(path+str(int(meters[k]))+".csv")




