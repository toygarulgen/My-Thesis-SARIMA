import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# from numpy import asarray
import pandas as pd
import requests
import json
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from entsoe import EntsoePandasClient
import statsmodels.api as sm
from math import sqrt
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
from numpy import asarray
from numpy import savetxt
import scipy.stats


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
#%%%###API#############
def EPIAS_API():
    down = './test.json'
    url = 'https://seffaflik.epias.com.tr/transparency/service/market/day-ahead-mcp?endDate=2019-12-31&startDate=2017-01-01'
    outpath=down
    generatedURL=url
    response = requests.get(generatedURL)
    if response.status_code == 200:
        with open(outpath, "wb") as out:
            for chunk in response.iter_content(chunk_size=128):
                out.write(chunk)
    with open(down) as json_file:
        data = json.load(json_file)
    body=data.get('body')
    gen=body.get('dayAheadMCPList')
    df=pd.DataFrame(gen)
    return(df)

#%%#############ENTSOE-API####################
def ENTSOE_API():
    client = EntsoePandasClient(api_key="2c958a88-3776-4f01-82cd-c957fdc4dc6a")

    country_code = 'EE', 'PT', 'ES', 'FR', 'FI', 'HU', 'SI', 'LV', 'NL', 'GR', 'BE'

    start = [pd.Timestamp('2016-12-31T22:00Z'), pd.Timestamp('2017-01-01T00:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T22:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T22:00Z'), pd.Timestamp('2016-12-31T23:00Z'), pd.Timestamp('2016-12-31T22:00Z'), pd.Timestamp('2016-12-31T23:00Z')]
    end= [pd.Timestamp('2019-12-31T21:00Z'), pd.Timestamp('2020-01-01T00:00Z'), pd.Timestamp('2019-12-31T22:00Z'), pd.Timestamp('2019-12-31T22:00Z'), pd.Timestamp('2019-12-31T21:00Z'), pd.Timestamp('2019-12-31T23:00Z'), pd.Timestamp('2019-12-31T23:00Z'), pd.Timestamp('2019-12-31T21:00Z'), pd.Timestamp('2019-12-31T23:00Z'), pd.Timestamp('2019-12-31T21:00Z'), pd.Timestamp('2019-12-31T22:00Z')]

    df1=[]
    iteration2=0
    ElectricityPrice=[]
    for iiii in range(len(country_code)):
        ElectricityPrice=client.query_day_ahead_prices(country_code[iteration2], start=start[iteration2], end=end[iteration2])
        if iiii==0:
            df1=pd.DataFrame({country_code[iteration2]:ElectricityPrice.values})
            iteration2=iteration2+1
            print(df1)
        else:
            df1[country_code[iteration2]]=pd.DataFrame({country_code[iteration2]:ElectricityPrice.values})
            iteration2=iteration2+1
            print(df1)
        # print(len(ElectricityPrice))
    return(df1)
#%%############SPLIT TRAIN TEST MODEL#############
df = ENTSOE_API()
df1 = EPIAS_API()
df1 = df1['priceEur']

countrycode = 'TR'
df[countrycode]=pd.DataFrame({countrycode:df1.values})

size = int(len(df) * 0.80)
train, test = df[0:size], df[size:len(df)]


# def adf_test(timeseries):
#     #Perform Dickey-Fuller test:
#     # print ('Results of Dickey-Fuller Test:')
#     dftest = adfuller(timeseries, autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#     for key,value in dftest[4].items():
#         dfoutput['Critical Value (%s)'%key] = value
    # print (dfoutput)
# print(adf_test(train))



###########PLOT##########
# count=1
# x = np.arange(0, 26280)
# newcountrycode = 'EE', 'PT', 'ES', 'FR', 'FI', 'HU', 'SI', 'LV', 'NL', 'GR', 'BE', 'TR'
# countriesnames = 'Estonia', 'Portugal', 'Spain', 'France', 'Finland', 'Hungary', 'Slovenia', 'Latvia', 'Netherlands', 'Greece', 'Belgium', 'Turkey'
# plt.figure(figsize = (15, 10), constrained_layout=True)
# for iteration3 in range(len(df.columns)):
#     plt.subplot(4, 4, count)
#     plt.plot(x, df[newcountrycode[iteration3]])
#     plt.title(countriesnames[iteration3], fontsize=12)
#     plt.xlabel('Hours', fontsize=10)
#     plt.ylabel('Forecasted Prices', fontsize=10)
#     count=count+1
# plt.tight_layout
# plt.savefig('plot1.png', dpi=1200)
# plt.savefig('plot1.eps', dpi=1200)
# plt.show()

# x = np.arange(0, 26280)
# newcountrycode = 'EE', 'PT', 'ES', 'FR', 'FI', 'HU', 'SI', 'LV', 'NL', 'GR', 'BE', 'TR'
# countriesnames = 'Estonia', 'Portugal', 'Spain', 'France', 'Finland', 'Hungary', 'Slovenia', 'Latvia', 'Netherlands', 'Greece', 'Belgium', 'Turkey'
# fig, ax = plt.subplots(3, 4, figsize=(25,15))
# iter1=0
# for iteration4 in range(3):
#     for iteration5 in range(4):
#         ax[iteration4, iteration5].plot(x, df[newcountrycode[iter1]], 'C1', linewidth=1)
#         ax[iteration4, iteration5].set_title(countriesnames[iter1], fontsize=20)
#         ax[iteration4, iteration5].set_xlabel('Hours', fontsize=12)
#         ax[iteration4, iteration5].set_ylabel('Forecasted Prices', fontsize=12)
#         iter1=iter1+1
# fig.tight_layout()
# plt.savefig('plot1.png', dpi=1200)
# plt.savefig('plot1.eps', dpi=1200)
# plt.show()

###########PLOT##########
#####PACF-PLOT######
# newcountrycode = 'EE', 'PT', 'ES', 'FR', 'FI', 'HU', 'SI', 'LV', 'NL', 'GR', 'BE', 'TR'
# countriesnames = 'Estonia', 'Portugal', 'Spain', 'France', 'Finland', 'Hungary', 'Slovenia', 'Latvia', 'Netherlands', 'Greece', 'Belgium', 'Turkey'
# fig, ax = plt.subplots(3, 4, figsize=(25,15))
# iteration3=0
# for iteration4 in range(3):
#     for iteration5 in range(4):
#         sm.graphics.tsa.plot_pacf(df[newcountrycode[iteration3]].squeeze(), lags=40, ax=ax[iteration4, iteration5])
#         ax[iteration4, iteration5].set_title('PACF: '+countriesnames[iteration3], fontsize=20)
#         iteration3=iteration3+1
# fig.tight_layout()
# # plt.savefig('PACF.png', dpi=1200)
# plt.savefig('PACF.eps', dpi=1200)
# plt.show()
# #####ACF-PLOT######
# fig, ax = plt.subplots(3, 4, figsize=(25,15))
# iteration3=0
# for iteration4 in range(3):
#     for iteration5 in range(4):
#         sm.graphics.tsa.plot_acf(df[newcountrycode[iteration3]].squeeze(), lags=40, ax=ax[iteration4, iteration5])
#         ax[iteration4, iteration5].set_title('ACF: '+countriesnames[iteration3], fontsize=20)
#         iteration3=iteration3+1
# fig.tight_layout()
# # plt.savefig('ACF.png', dpi=1200)
# plt.savefig('ACF.eps', dpi=1200)
# plt.show()
##########PLOT#########
#%%################ARIMA####################
TahminARIMA = pd.DataFrame(columns=df.columns).fillna(0)
newcountrycode = 'EE', 'PT', 'ES', 'FR', 'FI', 'HU', 'SI', 'LV', 'NL', 'GR', 'BE', 'TR'
countriesnames = 'Estonia', 'Portugal', 'Spain', 'France', 'Finland', 'Hungary', 'Slovenia', 'Latvia', 'Netherlands', 'Greece', 'Belgium', 'Turkey'
fig, ax = plt.subplots(3, 4, figsize=(25,15))
RMSEresult = list()
iter1=0
iteration4=0
iteration5=0
for iterationnew in range(len(newcountrycode)):

    mod = [sm.tsa.statespace.SARIMAX(train['EE'], trend='n', order=(1,0,1), seasonal_order=(2,0,0,24), enforce_invertibility=True),
          sm.tsa.statespace.SARIMAX(train['PT'], trend='n', order=(3,0,2), seasonal_order=(2,1,0,24)),
          sm.tsa.statespace.SARIMAX(train['ES'], trend='n', order=(3,0,2), seasonal_order=(2,1,0,24)),
          sm.tsa.statespace.SARIMAX(train['FR'], trend='n', order=(3,0,2), seasonal_order=(2,1,0,24)),
          sm.tsa.statespace.SARIMAX(train['FI'], trend='n', order=(1,0,1), seasonal_order=(2,0,0,24)),
          sm.tsa.statespace.SARIMAX(train['HU'], trend='n', order=(5,0,1), seasonal_order=(2,1,0,24)),
          sm.tsa.statespace.SARIMAX(train['SI'], trend='n', order=(2,0,2), seasonal_order=(2,1,0,24)),
          sm.tsa.statespace.SARIMAX(train['LV'], trend='n', order=(1,0,0), seasonal_order=(2,1,0,24)),
          sm.tsa.statespace.SARIMAX(train['NL'], trend='n', order=(2,0,1), seasonal_order=(2,1,0,24)),
          sm.tsa.statespace.SARIMAX(train['GR'], trend='n', order=(4,0,3), seasonal_order=(2,0,0,24)),
          sm.tsa.statespace.SARIMAX(train['BE'], trend='n', order=(4,0,3), seasonal_order=(2,0,0,24)),
          sm.tsa.statespace.SARIMAX(train['TR'], trend='n', order=(5,0,1), seasonal_order=(2,1,0,24))]

    results = mod[iterationnew].fit()
    print(results.summary())
    predictedvalue  = results.get_prediction(start = 21024, end= 21047, dynamic=True)
    arimaprediction = predictedvalue.predicted_mean
    
    TahminARIMA[newcountrycode[iterationnew]] = arimaprediction
    conf=predictedvalue.conf_int(alpha=0.05)

    RMSEresult.append(rmse(test[newcountrycode[iterationnew]].loc[21024:21047], arimaprediction.loc[21024:21047]))
    
    x = np.arange(0, 24)
    ax[iteration4, iteration5].plot(x, arimaprediction.loc[21024:21047], 'r', linewidth=3, label = "ARIMA Method Price")
    ax[iteration4, iteration5].plot(x, test[newcountrycode[iterationnew]].loc[21024:21047], 'k', linewidth=3, label="Actual Price")
    ax[iteration4, iteration5].fill_between(x, conf.iloc[:, 0], conf.iloc[:, 1], color='#ff7823', alpha=0.2, label="Confidence Interval (95%)")
    ax[iteration4, iteration5].set_title(countriesnames[iter1], fontsize=20)
    ax[iteration4, iteration5].set_xlabel('Hours', fontsize=12)
    ax[iteration4, iteration5].set_ylabel('Prices (Euro/MWh)', fontsize=12)
    ax[iteration4, iteration5].legend(loc="best")
    iter1 = iter1 + 1
    iteration5 = iteration5 + 1
    if iteration5 == 4:
        iteration4 = iteration4 + 1
        iteration5 = 0
fig.tight_layout()
plt.savefig('ARIMA.png', dpi=1200)
plt.savefig('ARIMA.eps', dpi=1200)
plt.show()







    
