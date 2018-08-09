#Load librairies
import numpy as np
import pandas as pd
import datetime
import keras
import numpy as np
import matplotlib.pyplot as plt
import datetime
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from scipy.cluster.hierarchy import dendrogram, linkage
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
#read an excel file
df=pd.read_excel("G:/Users/user/Desktop/prog python/pshforc/charge semi horaire.xlsx")
df.head()
df.info()
df.tail(3)
df.index
df.columns
df.describe()
#append day month , year and day of week 
df['T(°C)']=pd.to_numeric(df['T(°C)'],errors='coerce')
df['year'] = pd.DatetimeIndex(df['Date_Time']).year
df['month'] = pd.DatetimeIndex(df['Date_Time']).month
df['day'] = pd.DatetimeIndex(df['Date_Time']).day
df['weekday'] = pd.DatetimeIndex(df['Date_Time']).dayofweek
df.head()
# correlation matrix
correlation_matrix = df.corr()
correlation_matrix 
#convert dataframe to matrix
predictors =df.as_matrix()
#extract variable to predict
target=predictors[:,1]
#extract variables used as inputs
predictors=predictors[:,[2,3,4,5]]
#develop models with keras
ncols=predictors.shape[1]
model=Sequential()
model.add(Dense(100,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error' ,optimizer='adam')
model.fit(predictors, target,validation_split=0.3)

model.save('model.h5')
my_model=load_model('model.h5')
data=np.array([[15,2017,1,1]])
predictions=model.predict(data)
#performing an hierachical clustering 
df1=df[['P(MW)','T(°C)','year','month','day','weekday']]
Z = linkage(df1, 'ward')

tsp = pd.Series(df['P(MW)'],index=date)
tst =pd.Series(df['T(°C)'],index=date)
plt.figure()
tsp.plot()
tst.plot()
