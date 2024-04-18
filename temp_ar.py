import pandas as pd
from sklearn.impute import SimpleImputer
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # process the data
    temperature_data = pd.read_csv("data/temperature.csv").drop(columns=["datetime"])
    imputer = SimpleImputer(strategy='mean')
    imputed_data = pd.DataFrame(imputer.fit_transform(temperature_data))
    imputed_data.columns = temperature_data.columns
    imputed_data.index = temperature_data.index

    # training split, 70:15:15
    split1 = int(len(temperature_data) * 70 / 100)
    split2 = int(len(temperature_data) * 85 / 100)
    train_data = imputed_data.iloc[0:split1]
    val_data = imputed_data.iloc[split1:split2]
    test_data = imputed_data.iloc[split2:]

    # test different lags on validation set
    errors = []
    for lags in range(80):
        err_sum = 0

        for city in imputed_data.columns:
            # create model
            model = AutoReg(train_data[city], lags)
            model = model.fit()

            # compute error on validation set
            predictions = model.predict(start=len(train_data), end=len(train_data) + len(val_data)-1)
            err_sum += mean_squared_error(val_data[city], predictions)

        error = err_sum / len(imputed_data.columns)
        errors.append(error)
    plt.plot(errors, label="Mean Squared Error")
    plt.xlabel("Lags")
    plt.ylabel("Mean Squared Error")
    plt.title("Lags vs Mean Squared Error for Temperature AR")
    plt.savefig("ar_plots/temp_ar.jpg")