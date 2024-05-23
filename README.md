# Stream-Flow-Analysis

This project contains a script for analyzing streamflow data from the National Water Information System (NWIS). The script retrieves streamflow data for a specified sensor and period, calculates the total water volume for the current and past 9 years, identifies years with the highest and lowest flow, and generates a plot showing historical high and low flow data. The script also performs a linear regression prediction for future discharge values based on the current year's data.

## Files

- `StreamFlow_Analysis.py`: The main script to analyze and visualize streamflow data.

## Description

The `StreamFlow_Analysis.py` script retrieves streamflow data from the NWIS for a specified sensor and period. It calculates the total water volume for the current and past 9 years, identifies years with the highest and lowest flow, and generates a plot showing historical high and low flow data. The script also performs a linear regression prediction for future discharge values based on the current year's data.

## Dataset Information

The dataset used for this analysis is retrieved from the National Water Information System (NWIS). The specific sensor used in this analysis is the TRINITY R NR BURNT RANCH sensor, with sensor ID `11527000`.

## Hardcoded Dates

The script uses the following hardcoded dates for the analysis:
- `start_date`: '2023-5-21'
- `end_date`: '2023-6-13'

## Usage

### Prerequisites

Make sure you have the following Python libraries installed on your system:

- `hydrofunctions`
- `pandas`
- `matplotlib`
- `numpy`

You can install them using `pip`:

```sh
pip install hydrofunctions pandas matplotlib numpy
