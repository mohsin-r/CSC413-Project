import pandas as pd
from sklearn.impute import SimpleImputer
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# process the data
temp_data = pd.read_csv("data/temperature.csv")['Toronto'].to_frame()
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(temp_data)

# training split
split = int(len(temp_data) * 80 / 100)
train_data = imputed_data[:split]
test_data = imputed_data[split:]

# create model
lags = 3 # error doesn't change too much, but slowly decreases with more lags
model = AutoReg(train_data, lags)
model_fit = model.fit()

predictions = model_fit.predict(start=lags, end=len(train_data)-1)#start=len(train_data), end=(len(train_data) + len(test_data) - 1))

# compute error
mse = mean_squared_error(train_data[lags:], predictions)
print(f'Lags: {lags}, MSE: {mse}')