# Import statements
import pandas as pd
import numpy as np
from prophet import Prophet

# import data
data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/econ8310-assignment1/main/assignment_data_train.csv")

# grab the two needed columns
trips_data = data[['Timestamp','trips']]

# convert to datetime
trips_data['Timestamp'] = pd.to_datetime(trips_data["Timestamp"])

# Recreate the data frame with correct labels
trips_data = pd.DataFrame(trips_data.values, columns = ['ds','y'])


# Initialize Prophet instance; seasonality mode; add yearly/weekly/daily seasonality
# documentation: https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#prior-scale-for-holidays-and-seasonality
model = Prophet(changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality =True,
                seasonality_prior_scale=10.0)

# Adding monthly seasonality
# documentation: https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html
# but really just copying and trying everything from the Prophet website at this point
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# fit to data
modelFit = model.fit(trips_data)

# Create timeline for 744 hours in future
# documentation: https://facebook.github.io/prophet/docs/non-daily_data.html#sub-daily-data  
future = modelFit.make_future_dataframe(periods=744,freq='H')

# Generate predictions based on that timeline
forecast = modelFit.predict(future)

# convert to list that contains last 744
pred = forecast['trend'][-744:].to_list()