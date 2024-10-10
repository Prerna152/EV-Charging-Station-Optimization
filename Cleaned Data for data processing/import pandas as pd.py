import pandas as pd
import folium

from folium.plugins import HeatMap


# Load your data (charging stations, population density, and traffic forecast)

charging_stations = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\charging_stations.csv')

population_density = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\population_density.csv')

traffic_forecast = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\traffic_forecast.csv')


# Create a map centered on Amsterdam (or a relevant location)

map = folium.Map(location=[52.3676, 4.9041], zoom_start=12)


# Plot EV Charging stations (latitude, longitude)

charging_stations.apply(lambda row: folium.Marker([row['Latitude'], row['Longitude']], 

                                                  popup=row['Station Name']).add_to(map), axis=1)


                                                  # Drop rows where Latitude, Longitude, or population are NaN

population_density = population_density.dropna(subset=['Latitude', 'Longitude', 'population'])


# Create the heatmap again

pop_density_heatmap = HeatMap(list(zip(population_density['Latitude'], population_density['Longitude'], population_density['population'])))



# Create HeatMap for population density (if you have lat/lng for population data)

pop_density_heatmap = HeatMap(list(zip(population_density['Latitude'], population_density['Longitude'], population_density['population'])))

map.add_child(pop_density_heatmap)


# Create HeatMap for traffic flow

traffic_heatmap = HeatMap(list(zip(traffic_forecast['LAT'], traffic_forecast['LNG'], traffic_forecast['24 hours'])))
map.add_child(traffic_heatmap)


# Save the map

map.save(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\EV_station_heatmap.html')

import os

print(os.getcwd())


map.save(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\EV_station_heatmap.html')



print(population_density[['Latitude', 'Longitude', 'population']].head())

# Add a marker to check if the map displays

folium.Marker([52.3676, 4.9041], popup='Amsterdam').add_to(map)

map = folium.Map(location=[52.3676, 4.9041], zoom_start=12)



# Visualize the distribution of existing charging stations
import folium
import pandas as pd


# Load EV charging station data

charging_stations = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\charging_stations.csv')


# Create a map centered on the area of interest

map_ev = folium.Map(location=[52.3676, 4.9041], zoom_start=12)  # Example: Amsterdam


# Plot the charging stations

for _, row in charging_stations.iterrows():

    folium.Marker([row['Latitude'], row['Longitude']], popup=row['Station Name']).add_to(map_ev)


# Save the map to an HTML file

map_ev.save(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\EV_station_map.html')



##Exploratory Data Analysis (EDA):

# Population Density Heatmap Using Folium:Population Heatmap: Highlights densely populated areas where more EV users might live, leading to higher charging demand
import folium

from folium.plugins import HeatMap
import pandas as pd


# Load population density data

population_density = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\population_density.csv')


# Drop rows with NaN values in Latitude, Longitude, or population columns

population_density = population_density.dropna(subset=['Latitude', 'Longitude', 'population'])


# Create the heatmap

pop_density_map = folium.Map(location=[52.3676, 4.9041], zoom_start=12)


# Add heatmap layer

HeatMap(list(zip(population_density['Latitude'], population_density['Longitude'], population_density['population']))).add_to(pop_density_map)


# Save the map

pop_density_map.save(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\population_density_heatmap.html')




#2. Traffic Volume Heatmap Using Folium:hows areas with heavy vehicle activity, where EV drivers are likely to travel and potentially need charging infrastructure

# Load traffic volume data

traffic_forecast = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\traffic_forecast.csv')


# Create a map

traffic_map = folium.Map(location=[52.3676, 4.9041], zoom_start=12)


# Add heatmap layer for traffic volume

HeatMap(list(zip(traffic_forecast['LAT'], traffic_forecast['LNG'], traffic_forecast['24 hours']))).add_to(traffic_map)


# Save the map

traffic_map.save(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\traffic_volume_heatmap.html')



#3. Alternative Using Seaborn (for non-geospatial heatmaps):

import seaborn as sns

import matplotlib.pyplot as plt


# Create a pivot table for population density

heatmap_data = population_density.pivot_table(index="Latitude", columns="Longitude", values="population")


# Plot the heatmap

plt.figure(figsize=(10, 8))

sns.heatmap(heatmap_data, cmap="YlGnBu")

plt.show()



## Demand Modeling:



# Weights for each factor (can be adjusted)

population_weight = 0.4

traffic_weight = 0.3

charging_station_weight = 0.3


# Adjusted EV Vehicles estimation based on population, traffic, and charging stations

df['Adjusted EV Vehicles'] = (population_weight * df['population_density']) + \

                             (traffic_weight * df['traffic_forecast']) + \

                             (charging_station_weight * df['charging_stations'])


# Calculate EV Adoption Rate

df['EV Adoption Rate'] = df['Adjusted EV Vehicles'] / df['population_density']

print(df)




#merge the population density, traffic volume, and charging station data

# Load data

charging_stations = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\charging_stations.csv')

population_density = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\population_density.csv')

traffic_forecast = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Cleaned Data for data processing\traffic_forecast.csv')



# Merge datasets on common column (e.g., 'street_name')

merged_data = pd.merge(population_density, traffic_forecast, on='street_name', how='left')

merged_data = pd.merge(merged_data, charging_stations, on='street_name', how='left')


# Save the merged data to a CSV file

merged_data.to_csv(r'C:\Users\navin\OneDrive\Desktop\Final Project\Merged_EV_Data.csv', index=False)


print("Merged data has been successfully saved to the CSV file.")

# Now you can calculate Adjusted EV Vehicles


# Save the merged data to a CSV file

merged_data.to_csv(r'C:\Users\navin\OneDrive\Desktop\Final Project\Merged_EV_Data.csv', index=False)


print("Merged data has been successfully saved to the CSV file.")


def jls_extract_def(pandas, columns, x, on, how, suffixes, index):
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
    return merged_data


merged_data = jls_extract_def(pandas, columns, x, on, how, suffixes, index)


import pandas as pd

# Load the dataset (adjust the file path as needed)
file_path = r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\cleaned_no_duplicates.csv'
df = pd.read_csv(file_path)

# Define the coefficients for the demand model (adjust these based on your analysis)
alpha = 0.5  # Coefficient for Population Density
beta = 0.4   # Coefficient for Traffic Volume
gamma = 0.3  # Coefficient for Current Charging Stations

# Ensure that the column names for Population Density, Traffic Volume, and Charging Stations match your dataset
# Example column names:
# - Population Density = 'population'
# - Traffic Volume = '24 hours'
# - Charging Stations = 'Quantity'

# Calculate Charging Station Demand
df['Charging_Station_Demand'] = (alpha * df['population'] +
                                 beta * df['24 hours'] -
                                 gamma * df['Quantity'])

# Save the updated DataFrame with the demand model to a new CSV file
output_file_path = r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\charging_station_demand.csv'
df.to_csv(output_file_path, index=False)

print(f"Charging station demand model calculated and saved to {output_file_path}")
