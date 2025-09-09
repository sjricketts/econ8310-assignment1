# libraries
from statsmodels.tsa.api import ExponentialSmoothing
import pandas as pd

# import data
data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/econ8310-assignment1/main/assignment_data_train.csv")

# dependant variable
trips = data['trips']

# valid model
model = ExponentialSmoothing(trips,
                            trend='add' ,
                            seasonal='add', 
                            seasonal_periods=12)

# model fit
modelFit = model.fit()

#predict
pred = modelFit.forecast(744)
