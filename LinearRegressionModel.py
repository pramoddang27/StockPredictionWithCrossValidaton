from ObatiningStockData import *
from main import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def create_linear_reg(df):
    lr = LinearRegression()

    # Remove the NaN values
    df.dropna(inplace=True)

    # Split the data into a training set and a testing set
    x = df[["Lag1", "Lag2", "Lag3", "Lag4"]]
    y = df["Today"]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    lr.fit(x_train, y_train)

    # Predict the values
    y_pred = lr.predict(x_test)
    return y_pred, y_test

# After fitting our model it’s time to evaluate it.
# Let’s calculate the training and testing mean absolute errors for our model.
# The mean absolute error is the average of the absolute differences between predictions and actual values.
def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # R2 score
    r2 = r2_score(y_test, y_pred)

    print("R2 Score: ", r2)
    print("Mean Squared Error: ", mse)
    print("Mean Absolute Error: ", mae)

    df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    df_pred.reset_index(inplace=True)
    return df_pred

# Now let’s plot the actual and predicted values to see how well our model is performing.
def plot_model(df_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(df_pred["Actual"], label="Actual")
    plt.plot(df_pred["Predicted"], label="Predicted")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.show()


