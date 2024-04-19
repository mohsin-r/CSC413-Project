import pandas as pd
from sklearn.impute import SimpleImputer
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    mse_errors = []
    mae_errors = []
    search = True # NOTE: set to false to disable graph generation
    if search:
        for lags in range(80):
            mse = 0
            mae = 0

            for city in imputed_data.columns:
                # create model
                model = AutoReg(train_data[city], lags)
                model = model.fit()

                # compute error on validation set
                predictions = model.predict(start=len(train_data), end=len(train_data) + len(val_data)-1)
                mse += mean_squared_error(val_data[city], predictions)
                mae += mean_absolute_error(val_data[city], predictions)

            mse = mse / len(imputed_data.columns)
            mae = mae / len(imputed_data.columns)
            mse_errors.append(mse)
            mae_errors.append(mae)

        # Plot MSE, MAE
        plt.plot(mse_errors, label="Mean Squared Error")
        plt.xlabel("Lags")
        plt.ylabel("Mean Squared Error")
        plt.title("Lags vs Mean Squared Error for Temperature AR(x)")
        plt.savefig("ar_plots/temp_ar_mse.jpg")
        plt.close()

        plt.plot(mae_errors, label="Mean Absolute Error")
        plt.xlabel("Lags")
        plt.ylabel("Mean Absolute Error")
        plt.title("Lags vs Mean Absolute Error for Temperature AR(x)")
        plt.ticklabel_format(useOffset=False)
        plt.savefig("ar_plots/temp_ar_mae.jpg")
        plt.close()
    
    # 21 lags is chosen as the final model, 
    # achieved an MSE of 89.37832817504722
    
    best_lags = 21
    test_MAE = 0
    test_MSE = 0
    for city in imputed_data.columns:
        # create model
        model = AutoReg(train_data[city], best_lags)
        model = model.fit()

        # compute error on validation set
        predictions = model.predict(start=len(train_data) + len(val_data), end=len(train_data) + len(val_data) + len(test_data) - 1)
        test_MAE += mean_absolute_error(test_data[city], predictions)
        test_MSE += mean_squared_error(test_data[city], predictions)

    test_MAE = test_MAE / len(imputed_data.columns)
    test_MSE = test_MSE / len(imputed_data.columns)
    print(f"TEMP AR({best_lags}): MSE={test_MSE}, MAE={test_MAE}")