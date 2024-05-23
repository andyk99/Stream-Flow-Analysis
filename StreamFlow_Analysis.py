import hydrofunctions as hf
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import matplotlib.dates as mdates
import sys

# ANSI escape sequences for colored output
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
ENDC = '\033[0m'

def calculate_water_volume_and_change(df):
    """
    Calculate total water volume over a given period using the midpoint Riemann Sum method.
    
    Parameters:
    df (DataFrame): DataFrame containing discharge data with a datetime index.
    
    Returns:
    float: Total water volume in cubic feet.
    """
    total_sum = 0.0
    for i in range(1, len(df)):
        delta_time = (df.index[i] - df.index[i-1]).total_seconds()
        mid_discharge = (df['Discharge'].iloc[i] + df['Discharge'].iloc[i-1]) / 2.0
        total_sum += mid_discharge * delta_time 
    return total_sum

# Check for correct number of command line arguments
#if len(sys.argv) < 3:
#    print("Usage: <start_date> <end_date>")
#    sys.exit(1)

# Hardcoded dates for this example
#start_date = sys.argv[1]
#end_date = sys.argv[2]
start_date = '2023-5-21'
end_date = '2023-6-13'
sensor_id = '11527000'
year, month, day = [int(part) for part in end_date.split('-')]

# Fetch current data from NWIS
nwis_data = hf.NWIS(sensor_id, 'iv', start_date, end_date)
current_df = nwis_data.df('discharge')[:]
current_df = current_df.rename(columns={'USGS:' + sensor_id + ":00060:00000": "Discharge"})

dataframes = []
total_volumes = []

# Fetch historical data for the past 9 years
for i in range(9):
    historic_year = year - 1 - i
    historic_start_date = f'{historic_year}-{start_date.split("-")[1]}-{start_date.split("-")[2]}'
    historic_end_date = f'{historic_year}-{end_date.split("-")[1]}-{end_date.split("-")[2]}'
    nwis_historic = hf.NWIS(sensor_id, 'iv', historic_start_date, historic_end_date)
    df_temp = nwis_historic.df('discharge')[:]
    df_temp = df_temp.rename(columns={'USGS:' + sensor_id + ":00060:00000": "Discharge"})
    total_volume = calculate_water_volume_and_change(df_temp)
    dataframes.append(df_temp)
    total_volumes.append(total_volume)
    print(f"{YELLOW}Total volume for {historic_year}: {total_volume} cubic feet{ENDC}")

# Identify years with the highest and lowest flow
mostflow_year = total_volumes.index(max(total_volumes))
leastflow_year = total_volumes.index(min(total_volumes))
print(f"{GREEN}Year with the highest flow: {dataframes[mostflow_year].index[0].year} (Total volume: {max(total_volumes)} cubic feet){ENDC}")
print(f"{BLUE}Year with the lowest flow: {dataframes[leastflow_year].index[0].year} (Total volume: {min(total_volumes)} cubic feet){ENDC}")

# Calculate typical flow and its standard deviation
typical_flow_avg = pd.concat(dataframes).groupby([idx.dayofyear for idx in pd.concat(dataframes).index]).mean()
flow_std = pd.concat(dataframes).groupby([idx.dayofyear for idx in pd.concat(dataframes).index]).std()
typical_upper = typical_flow_avg + 0.5 * flow_std
typical_lower = typical_flow_avg - 0.5 * flow_std

# Plot historical high and low flow data
plt.figure(figsize=(12, 8))
plt.plot([idx.replace(year=year) for idx in dataframes[mostflow_year].index[:-1]], dataframes[mostflow_year]['Discharge'][:-1], color='green', label=f'Highest: {dataframes[mostflow_year].index[0].year}', linewidth=1)
plt.plot([idx.replace(year=year) for idx in dataframes[leastflow_year].index[:-1]], dataframes[leastflow_year]['Discharge'][:-1], color='lightskyblue', label=f'Lowest: {dataframes[leastflow_year].index[0].year}', linewidth=1.5)

def LinearRegressionSlopeIntercept(x_values, y_values):
    """
    Calculate the slope and intercept for a linear regression.
    
    Parameters:
    x_values (array-like): Array of x values.
    y_values (array-like): Array of y values.
    
    Returns:
    tuple: Slope and intercept of the regression line.
    """
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    slope = np.sum((x_values - x_mean) * (y_values - y_mean)) / np.sum((x_values - x_mean)**2)
    intercept = y_mean - slope * x_mean
    return slope, intercept

# Get current year data for 2023
current_year_data = current_df[current_df.index.year == 2023]
start_of_year = pd.Timestamp(year, 1, 1)
current_year_data.index = pd.to_datetime(current_year_data.index)  # Convert index to datetime

# Plot typical flow with upper and lower bounds
latest_date = max([df.index.max() for df in dataframes])
typical_days = [pd.Timestamp(year, 1, 1) + pd.Timedelta(days=x - 2) for x in typical_flow_avg.index]
plt.plot(typical_days, typical_upper['Discharge'], color='gray', linewidth=0.5)
plt.plot(typical_days, typical_lower['Discharge'], color='gray', linewidth=0.5)
plt.fill_between(typical_days, typical_lower['Discharge'], typical_upper['Discharge'], color='lightgray', label='Typical Flow')

# Calculate rate of change of flow over the last two weeks
dx_start_date = current_df.index.max() - pd.Timedelta(weeks=2)
dx_data = current_df[dx_start_date:]
if not dx_data.empty:
    start_flow = dx_data.iloc[0]['Discharge']
    end_flow = dx_data.iloc[-1]['Discharge']
    interval = (dx_data.index[-1] - dx_data.index[0]).total_seconds()
    flow_derivative = (end_flow - start_flow) / interval
    BOLD = '\033[1m'
    ENDC = '\033[0m'
    print(f"{BOLD}Rate of change of flow between {start_date} and {end_date}: {flow_derivative} CFS{ENDC}")
else:
    print("Insufficient data in given period")

# Remove the last week from the 2023 data for adding prediction line
prediction_line_startdate = pd.to_datetime(end_date).tz_localize('UTC') - timedelta(weeks=1)
modified_current_year_data = current_year_data[current_year_data.index <= prediction_line_startdate]
plt.plot(modified_current_year_data.index, modified_current_year_data['Discharge'], color='black', label='Current', linewidth=1.5)

# Linear regression for future prediction
numeric_dates = np.array([date.toordinal() for date in current_year_data.index])
slope, intercept = LinearRegressionSlopeIntercept(numeric_dates, current_year_data['Discharge'].values)

# Predict future discharge values
future_dates = pd.date_range(start=prediction_line_startdate, periods=9, freq='D')
future_numeric_dates = np.array([date.toordinal() for date in future_dates])
future_discharge = intercept + slope * future_numeric_dates
future_dates_plt = [pd.Timestamp.fromordinal(int(ordinal)) for ordinal in future_numeric_dates]
plt.plot(future_dates_plt, future_discharge, color='black', linewidth=1.5)
plt.axvline(pd.to_datetime(prediction_line_startdate), color='red', linestyle='--')

# Final plot adjustments
plt.xlabel('Date')
plt.ylabel('Discharge (CFS)')
plt.title('Streamflow Data Over Time by Year')
plt.legend(loc='upper right')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()
plt.show()