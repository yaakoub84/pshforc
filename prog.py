import numpy as np
import pandas as pd
import datetime
import keras
import numpy as np
import matplotlib.pyplot as plt
import datetime

df=pd.read_excel("G:/Users/user/Desktop/prog python/charge semi horaire.xlsx")
df.head()
df.info()
df.tail(3)
df.index
df.columns
df.describe()
df['T(°C)']=pd.to_numeric(df['T(°C)'],errors='coerce')
df['year'] = pd.DatetimeIndex(df['Date_Time']).year
df['month'] = pd.DatetimeIndex(df['Date_Time']).month
df['day'] = pd.DatetimeIndex(df['Date_Time']).day
df['weekday'] = pd.DatetimeIndex(df['Date_Time']).dayofweek
df.head()
correlation_matrix = df.corr()
correlation_matrix 

tsp = pd.Series(df['P(MW)'],index=date)
tst =pd.Series(df['T(°C)'],index=date)
plt.figure()
tsp.plot()
tst.plot()
