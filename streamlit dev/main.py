import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
import pulp  
from PIL import Image

# Load your datasets
df_stations = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Charging_Station_Cleaned.csv')  # Dataset for existing charging stations
df_demand = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\normalized_population_traffic_with_demand.csv')  # Dataset for demand model
df_optimization = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\cluster_centroids.csv')  # Optimized locations

# Title for your Streamlit app
st.title("EV Charging Station Optimization Project")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select Section", ["EDA", "Demand Modeling", "Optimization", "Validation", "Reporting"])

### Section 1: Exploratory Data Analysis (EDA)
if options == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Distribution of Existing Charging Stations")
    # Plot the existing charging stations on a map
    fig_stations = px.scatter_mapbox(df_stations, lat="Latitude", lon="Longitude", hover_name="Address",
                                     zoom=10, height=500, color="Quantity")  # Ensure 'Quantity' exists
    fig_stations.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_stations)

    st.subheader("Heatmap: Population Density and Traffic Volume")

    # Clean column names to avoid issues with spaces or cases
    df_demand.columns = df_demand.columns.str.strip().str.lower()

    # Step 1: Filter data for the year 2025 only
    df_demand = df_demand[df_demand['year'] == 2025]

     #Step 2: Display summary statistics for population density and traffic volume
    #st.write("Summary statistics for Population Density and Traffic Volume (2025):")
    #st.write(df_demand[['population density', 'traffc_volume']].describe())

    # Add a slider to filter data based on population density and traffic volume
    population_threshold = st.slider("Minimum Population Density", min_value=0, max_value=int(df_demand['population density'].max()), value=int(df_demand['population density'].median()))
    traffic_threshold = st.slider("Minimum Traffic Volume", min_value=0, max_value=int(df_demand['traffc_volume'].max()), value=int(df_demand['traffc_volume'].median()))

    # Filter the data based on selected thresholds
    filtered_data = df_demand[(df_demand['population density'] >= population_threshold) & (df_demand['traffc_volume'] >= traffic_threshold)]

    # Step 3: Check if filtered data has rows
    #st.write(f"Number of rows after filtering: {filtered_data.shape[0]}")

    if filtered_data.empty:
        st.write("No data available for the selected filters.")
    else:
        # Step 4: Create the heatmap
        fig = px.density_mapbox(
            filtered_data, 
            lat="latitude_population", 
            lon="longitude_population", 
            z="population density",  # Color the heatmap by population density
            radius=10,  # Adjust the radius for better visual effects
            zoom=10, 
            mapbox_style="open-street-map",
            color_continuous_scale=px.colors.sequential.Viridis,  # Use a color scale
            title="Heatmap: Population Density and Traffic Volume"
        )

        # Display the heatmap
        st.plotly_chart(fig)


### Section 2: Demand Modeling
if options == "Demand Modeling":
    st.header("Demand Modeling")
    st.subheader("High-Demand Areas")

    st.write("We identified high-demand areas based on population density, EV vehicles, and traffic volume.")
    
    # Ensure the correct column names are used for lat, lon, size, and color
    fig_demand_model = px.scatter_mapbox(
        df_demand, 
        lat="latitude_population",  # Ensure this column exists
        lon="longitude_population",  # Ensure this column exists
        hover_name="street_name",    # Update if needed, ensure the correct column name
        size="Demand_Model",         # Use 'Demand_Model' or 'Normalized_Demand_Model'
        color="Demand_Model",        # Use the correct column for color, like 'Demand_Model'
        zoom=10
    )
    
    fig_demand_model.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_demand_model)



import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from geopy.geocoders import Nominatim

# Load the dataset
df = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\normalized_population_traffic_with_demand.csv')

# Clean column names to avoid issues with spaces or cases
df.columns = df.columns.str.strip().str.lower()

# Step 1: Filter data for the year 2025 at the backend (no visible year filter)
selected_year = 2025  # Hardcoded to filter for the year 2025
df_filtered = df[df['year'] == selected_year]

# Check if there is any data for the selected year
if df_filtered.empty:
    st.write(f"No data available for the year {selected_year}.")
else:
    # Step 2: Extract coordinates (latitude and longitude) and demand for clustering
    coords = df_filtered[['latitude_population', 'longitude_population']].values
    demand = df_filtered['demand_model'].values

    # Step 3: Apply K-Means clustering (you can adjust n_clusters)
    kmeans = KMeans(n_clusters=10, random_state=42)  # Adjust n_clusters as needed
    df_filtered['Cluster'] = kmeans.fit_predict(coords)

    # Step 4: Get the centroids of the clusters
    centroids = kmeans.cluster_centers_

    # Step 5: Reverse Geocoding to Get Addresses
    geolocator = Nominatim(user_agent="charging_station_locator")
    addresses = []
    for centroid in centroids:
        location = geolocator.reverse((centroid[0], centroid[1]), exactly_one=True)
        addresses.append(location.address if location else "Address not found")

    # Save centroids and addresses into a dataframe
    centroids_df = pd.DataFrame(centroids, columns=['latitude', 'longitude'])
    centroids_df['address'] = addresses

    # Step 6: Visualize the centroids on an interactive map with Plotly
    st.subheader(f"Potential New Charging Station Locations")

    fig = px.scatter_mapbox(
        centroids_df, 
        lat="latitude", 
        lon="longitude", 
        hover_name="address",  # Display the address when hovering
        zoom=10,
        size_max=15, 
        color_discrete_sequence=["red"],  # Use red for centroids
        title=f"Potential Charging Station Locations"
    )

    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

    # Display the centroids and their addresses in the app
    st.write("Potential new charging station locations (centroids with addresses):")
    st.write(centroids_df)


### Section 4: Validation (Scenario Analysis)
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\normalized_population_traffic_with_demand.csv')

# Clean column names to avoid issues with spaces or cases
df.columns = df.columns.str.strip().str.lower()

# Step 1: Filter data for the year 2025 only (hardcoded in the backend)
df = df[df['year'] == 2025]  # Only include data for 2025

# Step 2: Add a dropdown for selecting the scenario
scenario = st.selectbox("Select Scenario", ["Moderate Demand Growth (20%)", "High Traffic Volume Growth (30%)", "Shift in Traffic Patterns"])

# Step 3: Use the already calculated Normalized Demand Model for all scenarios and adjust it dynamically
if scenario == "Moderate Demand Growth (20%)":
    st.subheader("Scenario 1: Moderate Demand Growth (20% increase)")
    
    # Scenario 1: Moderate Demand Growth (20% increase)
    growth_rate = 1.20
    df['normalized_demand_model_scenario'] = df['normalized_demand_model'] * growth_rate

elif scenario == "High Traffic Volume Growth (30%)":
    st.subheader("Scenario 2: High Traffic Volume Growth (30% increase)")
    
    # Scenario 2: Apply traffic volume growth to the existing Normalized Demand Model
    traffic_growth_rate = 1.30
    df['normalized_demand_model_scenario'] = df['normalized_demand_model'] * traffic_growth_rate  # Simply adjust by traffic growth rate

elif scenario == "Shift in Traffic Patterns":
    st.subheader("Scenario 3: Shift in Traffic Patterns")
    
    # Scenario 3: Adjust traffic patterns using the already calculated Normalized Demand Model
    df['traffic_volume_scenario'] = df['traffc_volume']  # Start with the original traffic volume
    
    # Decrease traffic volume by 20% for low-demand areas (below 33rd percentile)
    df.loc[df['normalized_demand_model'] < df['normalized_demand_model'].quantile(0.33), 'normalized_demand_model_scenario'] = df['normalized_demand_model'] * 0.8  # Decrease by 20%
    
    # Increase traffic volume by 10% for high-demand areas (above 66th percentile)
    df.loc[df['normalized_demand_model'] > df['normalized_demand_model'].quantile(0.66), 'normalized_demand_model_scenario'] = df['normalized_demand_model'] * 1.1  # Increase by 10%

# Step 4: Handle missing or NaN values in the 'normalized_demand_model_scenario' column
df['normalized_demand_model_scenario'].fillna(0, inplace=True)  # Fill NaN values with 0 or another appropriate value

# Step 5: Apply K-Means clustering to simulate new locations for each scenario based on Normalized Demand
num_clusters = 20  # Simulate 20 new charging station locations

# Use KMeans to generate new latitude and longitude coordinates based on normalized demand
coords = df[['latitude_population', 'longitude_population']].values
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(coords)

# Get the centroids of the clusters (new potential locations for charging stations)
centroids = kmeans.cluster_centers_

# Create a new DataFrame for centroids (potential new charging station locations)
centroids_df = pd.DataFrame(centroids, columns=['latitude', 'longitude'])
centroids_df['normalized_demand'] = df.groupby('cluster')['normalized_demand_model_scenario'].mean().values  # Assign demand to each centroid

# Step 6: Remove any centroids with NaN or zero demand (if this is inappropriate for visualization)
centroids_df = centroids_df[centroids_df['normalized_demand'] > 0]  # Ensure no invalid values are used

# Step 7: Visualize the new potential locations on an interactive map using Plotly
st.subheader(f"Simulated Locations for New Charging Stations - {scenario}")

fig = px.scatter_mapbox(
    centroids_df, 
    lat="latitude", 
    lon="longitude", 
    hover_name="normalized_demand",  # Hover to show demand
    zoom=10,
    size="normalized_demand",  # Adjust point size based on normalized demand
    color="normalized_demand",  # Color points based on normalized demand
    size_max=15, 
    color_continuous_scale=px.colors.sequential.Viridis,  # Use color scale to differentiate demand
    title=f"Simulated Charging Station Locations for {scenario} (Normalized Demand)"
)

# Update the layout to show the map
fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig)


### Section 5: Reporting
if options == "Reporting":
    st.header("Final Reporting")
    
    st.write("Here is the final report showing the current and optimized locations of charging stations.")
    
    # Compare current and optimized locations of charging stations
    st.subheader("Current Charging Stations")
    fig_current_stations = px.scatter_mapbox(df_stations, lat="Latitude", lon="Longitude", hover_name="Station Name",
                                             zoom=10, color="Quantity")  # Make sure 'Quantity' exists
    fig_current_stations.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_current_stations)

    st.subheader("Optimized Charging Stations")
    fig_optimized_stations = px.scatter_mapbox(df_optimization, lat="latitude", lon="longitude", hover_name="station_name",
                                               color="demand_coverage", size="demand_coverage", zoom=10)
    fig_optimized_stations.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_optimized_stations)

    st.write("This report highlights the areas that are currently underserved and the optimized placement of charging stations.")


# Section 6 - Validation (Scenario Analysis)

# Section 6 - Validation (Scenario Analysis)
if options == "Validation":
    st.header("Scenario Analysis: Demand and Traffic Volume Growth")

    # Load the dataset
    df = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\normalized_population_traffic_with_demand.csv')

    # Clean column names to avoid issues with spaces or cases
    df.columns = df.columns.str.strip().str.lower()

    # Filter data for a specific year (e.g., 2025)
    year_filter = 2025
    df_filtered = df[df['year'] == year_filter]

    # Calculate metrics for the chosen year
    ev_vehicles_current = df_filtered['ev vehicles'].sum()
    population_density_current = df_filtered['population density'].mean()
    traffic_volume_current = df_filtered['traffc_volume'].sum()

    # Calculate metrics for the previous year (e.g., 2024)
    previous_year = year_filter - 1
    df_previous = df[df['year'] == previous_year]

    ev_vehicles_previous = df_previous['ev vehicles'].sum() if not df_previous.empty else 0
    population_density_previous = df_previous['population density'].mean() if not df_previous.empty else 0
    traffic_volume_previous = df_previous['traffc_volume'].sum() if not df_previous.empty else 0

    # Calculate the percentage change for each metric
    ev_vehicles_change = ((ev_vehicles_current - ev_vehicles_previous) / ev_vehicles_previous * 100) if ev_vehicles_previous != 0 else 0
    population_density_change = ((population_density_current - population_density_previous) / population_density_previous * 100) if population_density_previous != 0 else 0
    traffic_volume_change = ((traffic_volume_current - traffic_volume_previous) / traffic_volume_previous * 100) if traffic_volume_previous != 0 else 0

    # Display the metrics with arrows indicating upward/downward trends
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="EV Vehicles", value=f"{ev_vehicles_current:,}", 
                  delta=f"{ev_vehicles_change:.2f}%" if ev_vehicles_change != 0 else "0.00%", 
                  delta_color="inverse" if ev_vehicles_change < 0 else "normal")

    with col2:
        st.metric(label="Population Density", value=f"{population_density_current:.2f}", 
                  delta=f"{population_density_change:.2f}%" if population_density_change != 0 else "0.00%", 
                  delta_color="inverse" if population_density_change < 0 else "normal")

    with col3:
        st.metric(label="Traffic Volume", value=f"{traffic_volume_current:,}", 
                  delta=f"{traffic_volume_change:.2f}%" if traffic_volume_change != 0 else "0.00%", 
                  delta_color="inverse" if traffic_volume_change < 0 else "normal")

    # Select Scenario (Demand Growth or Traffic Growth)
    scenario = st.selectbox("Select Scenario", ["High Demand Growth", "Low Demand Growth", "High Traffic Volume Growth", "Low Traffic Volume Growth"])

    # Ask the user to choose the percentage for demand or traffic volume growth
    growth_percentage = st.slider("Select Growth Percentage", min_value=5, max_value=40, step=5, value=10)

    # Apply the selected scenario and percentage
    if scenario == "High Demand Growth":
        st.subheader(f"High Demand Growth ({growth_percentage}%)")
        df_filtered['demand_model_scenario'] = df_filtered['demand_model'] * (1 + (growth_percentage / 100))
        coverage_target = 0.9  # 90% coverage target for high demand
        color_scale = "Reds"
        size_scale = 10
        max_stations = int(20 + (growth_percentage / 5))
        st.write(f"In this scenario, we model a **{growth_percentage}% growth** in demand for EV charging stations.")
    
    elif scenario == "Low Demand Growth":
        st.subheader(f"Low Demand Growth ({growth_percentage}%)")
        df_filtered['demand_model_scenario'] = df_filtered['demand_model'] * (1 + (growth_percentage / 100))
        coverage_target = 0.8
        color_scale = "Blues"
        size_scale = 5
        max_stations = int(15 + (growth_percentage / 5))
        st.write(f"In this scenario, we model a **{growth_percentage}% growth** in demand for EV charging stations.")

    elif scenario == "High Traffic Volume Growth":
        st.subheader(f"High Traffic Volume Growth ({growth_percentage}%)")
        df_filtered['traffic_volume_scenario'] = df_filtered['traffc_volume'] * (1 + (growth_percentage / 100))
        beta, gamma = 0.2, 0.3
        df_filtered['demand_model_scenario'] = (beta * df_filtered['ev vehicles'] + gamma * df_filtered['traffic_volume_scenario'])
        coverage_target = 0.9
        color_scale = "Oranges"
        size_scale = 10
        max_stations = int(20 + (growth_percentage / 5))
        st.write(f"In this scenario, we model a **{growth_percentage}% growth** in traffic volume, increasing the demand for EV charging stations.")

    elif scenario == "Low Traffic Volume Growth":
        st.subheader(f"Low Traffic Volume Growth ({growth_percentage}%)")
        df_filtered['traffic_volume_scenario'] = df_filtered['traffc_volume'] * (1 + (growth_percentage / 100))
        beta, gamma = 0.2, 0.3
        df_filtered['demand_model_scenario'] = (beta * df_filtered['ev vehicles'] + gamma * df_filtered['traffic_volume_scenario'])
        coverage_target = 0.8
        color_scale = "Purples"
        size_scale = 5
        max_stations = int(15 + (growth_percentage / 5))
        st.write(f"In this scenario, we model a **{growth_percentage}% growth** in traffic volume, leading to slower demand growth for EV charging stations.")

    # Filter to the top 50% demand locations for optimization
    threshold = df_filtered['demand_model_scenario'].quantile(0.50)
    df_filtered = df_filtered[df_filtered['demand_model_scenario'] > threshold].reset_index(drop=True)

    # Optimization Problem
    num_locations = len(df_filtered)
    problem = pulp.LpProblem("EV_Charging_Station_Optimization", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("station", range(num_locations), cat='Binary')

    # Objective function: minimize the total cost
    costs = df_filtered['demand_model_scenario'].values
    problem += pulp.lpSum([costs[i] * x[i] for i in range(num_locations)])

    # Ensure the coverage of total demand (e.g., 90% or 80%)
    total_demand = df_filtered['demand_model_scenario'].sum()
    problem += pulp.lpSum([df_filtered['demand_model_scenario'][i] * x[i] for i in range(num_locations)]) >= coverage_target * total_demand

    # Dynamically limit the number of charging stations based on growth
    problem += pulp.lpSum([x[i] for i in range(num_locations)]) <= max_stations

    # Solve the problem
    problem.solve(pulp.PULP_CBC_CMD())

    # Extract results
    df_filtered['station_placement'] = [pulp.value(x[i]) for i in range(num_locations)]

    # Plot the results on a map (use only the placed stations)
    df_result = df_filtered[df_filtered['station_placement'] == 1]
    fig = px.scatter_mapbox(
        df_result,
        lat="latitude_population", 
        lon="longitude_population", 
        hover_name="street_name",  
        size="demand_model_scenario",  # Size of the point represents demand
        color="demand_model_scenario",  # Color represents demand level
        color_continuous_scale=color_scale,
        size_max=size_scale,
        zoom=10,
        height=600
    )

    # Set map style and display the map
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

    # Explanation section for Scenario
    st.subheader(f"Scenario Explanation: {scenario} ({growth_percentage}%)")
    st.write(f"The solution ensures that at least **{int(coverage_target * 100)}% of the total demand** is covered, with a maximum of {max_stations} new charging stations placed under the {scenario} scenario with {growth_percentage}% growth.")
