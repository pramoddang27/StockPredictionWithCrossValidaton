from ObatiningStockData import *
from LinearRegressionModel import *

import plotly.express as px

# Get the stock data
if __name__ == "__main__":
    symbol = "KPITTECH.NS"
    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime(2024, 2, 25)

    ftse_lags = create_lagged_series(symbol, start_date, end_date, lags=5)
    # print(ftse_lags.head(5))

    y_pred, y_test = create_linear_reg(ftse_lags)
    #print(y_pred)

    df_pred = evaluate_model(y_test, y_pred)
    print(df_pred.head(5))

    #fig = px.line(df_pred, x="Date", y=["Actual", "Predicted"],color="variable",
                  #title="Linear Regression Model: Actual Prices vs. Predicted Prices.")
    #fig.show()

    plot_model(df_pred)

