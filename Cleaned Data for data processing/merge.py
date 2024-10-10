import pandas as pd

# Load datasets
charging_stations = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\charging_stations.csv')
population_density = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\population_density.csv')
traffic_forecast = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\traffic_forecast.csv')

# Ensure column names have no trailing spaces and are consistent
population_density.columns = population_density.columns.str.strip()
traffic_forecast.columns = traffic_forecast.columns.str.strip()
charging_stations.columns = charging_stations.columns.str.strip()

# Rename the traffic forecast columns dynamically (for year-wise data)
# Example: '24 hours' becomes '24 hours_2024' for the year 2024 if it exists
traffic_forecast = traffic_forecast.rename(columns=lambda x: f"{x}_year" if x in ['24 hours', 'daylight', 'evening', 'night'] else x)

# Merge datasets on common 'street_name' column
merged_data = pd.merge(population_density, traffic_forecast, on='street_name', how='left', suffixes=('_pop', '_traffic'))
merged_data = pd.merge(merged_data, charging_stations, on='street_name', how='left', suffixes=('', '_charging'))

# Save the merged data to a CSV file
merged_data.to_csv(r'C:\Users\navin\OneDrive\Desktop\Final Project\Merged_EV_Data.csv', index=False)

print("Merged data has been successfully saved to the CSV file.")
