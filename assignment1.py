# libraries
from statsmodels.tsa.api import ExponentialSmoothing
import pandas as pd

# import data
data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/econ8310-assignment1/main/assignment_data_train.csv")

# dependant variable
model = data['trips']

# model fit with seasonality (not dampened)
modelFit = ExponentialSmoothing(model, 
                                trend='add' ,
                                seasonal='add', 
                                seasonal_periods=12).fit()
