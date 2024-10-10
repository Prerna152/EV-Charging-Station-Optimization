import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import plotly.express as px
import pulp
from sklearn.cluster import KMeans
import plotly.express as px
from geopy.geocoders import Nominatim


# Load your datasets
df_stations = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\Charging_Station_Cleaned.csv')  # Dataset for existing charging stations
df_demand = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\normalized_population_traffic_with_demand.csv')  # Dataset for demand model
df_optimization = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\cluster_centroids.csv')  # Optimized locations

# Sidebar for navigation and displaying app name
st.sidebar.title("EV Charging Station Optimization: Amsterdam")  # App name in the sidebar
options = st.sidebar.radio("Select Section", ["Introduction", "Technical Overview", "EDA", "Demand Modeling", "Optimization", "Validation", "Findings"])

# Section 1 - Introduction
if options == "Introduction":
    st.title("EV Charging Station Optimization: Amsterdam")  # App name in the sidebar

    # Adding an image of Amsterdam
    image_path = r"C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\AdobeStock.jpg"  # Use a raw string or double backslashes
    st.image(image_path, caption="Amsterdam - Leading the Charge in EV Adoption", use_column_width=True)

    # Create two columns: one for the introduction and one for the graph
    col1, col2 = st.columns([2, 1])  # Adjust the ratio of the columns

    # Introduction Section in the left column (col1)
    with col1:
        st.header("Introduction: Electric Vehicle Growth in Amsterdam")
        st.write("""
        Amsterdam is at the forefront of the electric vehicle (EV) revolution due to its unique urban landscape and commitment to sustainability. The Netherlands has seen over **340,000 fully electric vehicles** on the road as of 2023, a **200% increase** over the last five years. However, this surge in EV adoption presents challenges for the city’s charging infrastructure.
        """)

    # Graph in the right column (col2)
    with col2:
        # Data for EV growth
        data = {
            'Year': ['2019', '2020', '2021', '2022', '2023'],
            'EVs': [50000, 100000, 200000, 340000, 460000]  # Estimated number for 2023
        }

        # Matplotlib Line Graph
        fig, ax = plt.subplots()
        ax.plot(data['Year'], data['EVs'], marker='o', color='b')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of EVs')
        ax.set_title('Growth of Electric Vehicles in the Netherlands')
        # Display the chart in the second column
        st.pyplot(fig)

    # Additional content for introduction
    st.write("""
    ### Amsterdam is particularly suited for optimizing EV infrastructure because of its:
    - **Population density**: 4,391 people per km² in the city center.
    - **Projected EV Growth**: Expected to exceed **1.9 million EVs by 2030** nationwide, with a significant portion in Amsterdam.
    - **Current charging stations**: Amsterdam has over **3,000 public charging stations**, but more are needed to meet the growing demand.
    """)
    # Section 3: Problem Statement
    st.header("Problem Statement: Addressing the EV Charging Infrastructure Gap")
    st.write("""
    The rapid adoption of electric vehicles in Amsterdam has outpaced the development of charging infrastructure. As the number of EVs continues to rise, the city faces several challenges:

    - **Inadequate Charging Capacity**: While there are over **3,000 public charging stations**, they are not evenly distributed across the city, leaving many areas underserved.
    - **Forecasted Growth Pressure**: With projections indicating over **1.9 million EVs** by 2030 nationwide, Amsterdam will need to significantly expand its charging infrastructure to keep up with demand.
    - **Optimization Challenge**: Simply adding more charging stations isn’t enough. The challenge lies in **optimizing the locations** of new charging stations to maximize coverage, minimize congestion, and ensure accessibility for all EV owners.

    This project seeks to address these challenges through data analysis, demand modeling, and spatial optimization to provide an effective solution for Amsterdam's charging infrastructure.
    """)

    # Section 4: Impact
    st.header("Impact: Sustainable and Scalable EV Infrastructure")
    st.write("""
    A well-optimized charging infrastructure will have significant **economic**, **environmental**, and **social** benefits for Amsterdam:

    - **Economic Impact**: Efficient placement of charging stations can reduce costs by maximizing resource utilization and ensuring that infrastructure is built where it is most needed.
    - **Environmental Impact**: Reducing vehicle emissions and promoting the use of electric vehicles by making charging stations more accessible, especially in high-density areas.
    - **Social Impact**: Improving the quality of life for Amsterdam residents by reducing air pollution and providing equitable access to charging infrastructure in all neighborhoods.

    By strategically planning for future EV growth, Amsterdam can lead the way in sustainable urban mobility and remain a global role model for electric vehicle adoption.
    """)

# Section 2 - Technical Overview
elif options == "Technical Overview":
    st.header("Technical Summary")
    
    # Load and resize the image
    image_path = r"C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\technical.webp"  # Use a raw string or double backslashes
    img = Image.open(image_path)
    img_resized = img.resize((700, 300))  # Resize the image
    
    # Display the resized image in Streamlit
    st.image(img_resized, use_column_width=False)

    st.header("Data Insights")
    st.write("""
    Collected data on **traffic**, **population density**, and **EV charging stations** from sources like **Gemeente Amsterdam** and **Electromaps**. This provided a solid foundation to analyze future EV infrastructure needs.
    """)

    st.header("Demand Forecasting")
    st.write("""
    Developed a model to predict where new EV charging stations are most needed, based on **population growth**, **traffic volume**, and **EV adoption rates**.
    """)

    st.header("Optimized Station Placement")
    st.write("""
    Used **spatial optimization techniques** to strategically identify the best locations for new charging stations. This ensures **maximum coverage**.
    """)

    st.header("Business Impact")
    st.write("""
    Helps decision-makers prioritize where to build new stations, supporting **Amsterdam’s sustainability goals** by meeting growing EV demand and reducing user inconvenience.
    """)

# Section 3 - Exploratory Data Analysis (EDA)
elif options == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Distribution of Existing Charging Stations")
    # Plot the existing charging stations on a map
    fig_stations = px.scatter_mapbox(df_stations, lat="Latitude", lon="Longitude", hover_name="Address",
                                     zoom=10, height=500, color="Quantity", color_continuous_scale="Cividis")  # Ensure 'Quantity' exists
    fig_stations.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_stations)

    st.subheader("Heatmap: Population Density and Traffic Volume")

    # Clean column names to avoid issues with spaces or cases
    df_demand.columns = df_demand.columns.str.strip().str.lower()

    # Step 1: Filter data for the year 2025 only
    df_demand = df_demand[df_demand['year'] == 2025]

    # Step 2: Display summary statistics for population density and traffic volume
    # st.write("Summary statistics for Population Density and Traffic Volume (2025):")
    # st.write(df_demand[['population density', 'traffc_volume']].describe())

    # Add a slider to filter data based on population density and traffic volume
    population_threshold = st.slider("Minimum Population Density", min_value=0, max_value=int(df_demand['population density'].max()), value=int(df_demand['population density'].median()))
    traffic_threshold = st.slider("Minimum Traffic Volume", min_value=0, max_value=int(df_demand['traffc_volume'].max()), value=int(df_demand['traffc_volume'].median()))

    # Filter the data based on selected thresholds
    filtered_data = df_demand[(df_demand['population density'] >= population_threshold) & (df_demand['traffc_volume'] >= traffic_threshold)]

    # Step 3: Check if filtered data has rows
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

    # Add the explanation of the heatmap
    st.subheader("Explanation: Heatmap of Population Density and Traffic Volume")
    st.write("""
    This heatmap provides insights into two important factors that drive the demand for electric vehicle (EV) charging stations in Amsterdam:

    1. **Population Density**: Areas with higher population densities are likely to have more EV users due to the greater number of residents. These areas are represented by darker shades on the heatmap, indicating a higher concentration of people and therefore a higher potential demand for charging stations.

    2. **Traffic Volume**: The traffic volume, while not directly represented in this heatmap, correlates with population density and is also a major driver of demand for EV charging infrastructure. Areas with high population density often experience heavy traffic, which means more vehicles on the road and a greater need for charging facilities.

    The **darker regions** on the heatmap signify areas where both population density and traffic volume converge, indicating **high-priority locations** for deploying new charging stations. These regions should be the focus for future EV infrastructure development.
    """)


# Section 4 - Demand Modeling
elif options == "Demand Modeling":
    st.title("Demand Modeling")
    df_demand = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\normalized_population_traffic_with_demand.csv')
    df_demand_2025 = df_demand[df_demand['year'] == 2025]  # Ensure _year contains 2025 data

    fig_demand_model = px.scatter_mapbox(
        df_demand_2025, 
        lat="latitude_population",  
        lon="longitude_population",  
        hover_name="street_name",    
        size="Normalized_Demand_Model",  
        color="Normalized_Demand_Model",  
        zoom=10,
        color_continuous_scale="Cividis"  
    )
    fig_demand_model.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_demand_model)

    st.write("""
    The graph displays high-demand areas for new EV charging stations. Larger circles represent higher demand based on factors like population density and traffic volume, while darker colors indicate more intense demand. By analyzing the size and color, optimal spots for new charging stations can be identified. Hover over the circles for detailed location information.
    """)

# Section 5 - Optimization Model
elif options == "Optimization":
    st.header("Optimization Model")
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

        # Step 5: Reverse Geocoding to Get Addresses (move this outside the loop)
        geolocator = Nominatim(user_agent="charging_station_locator")
        addresses = []
        for centroid in centroids:
            try:
                location = geolocator.reverse((centroid[0], centroid[1]), exactly_one=True)
                addresses.append(location.address if location else "Address not found")
            except:
                addresses.append("Address not found")  # Handle geocoding failures gracefully

        # Save centroids and addresses into a dataframe
        centroids_df = pd.DataFrame(centroids, columns=['latitude', 'longitude'])
        centroids_df['address'] = addresses  # Now the lengths should match

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

# Section 6 - Validation (Scenario Analysis)
elif options == "Validation":
    st.header("Scenario Analysis: Demand and Traffic Volume Growth")

    # Load the dataset
    df = pd.read_csv(r'C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\normalized_population_traffic_with_demand.csv')
    df = df[df['year'] == 2025]  # Ensure '_year' contains 2025 data

    # Select Scenario (Demand Growth or Traffic Growth)
    scenario = st.selectbox("Select Scenario", ["High Demand Growth", "Low Demand Growth", "High Traffic Volume Growth", "Low Traffic Volume Growth"])

    # Ask the user to choose the percentage for demand or traffic volume growth
    growth_percentage = st.slider("Select Growth Percentage", min_value=5, max_value=40, step=5, value=10)

    # Apply the selected scenario and percentage
    if scenario == "High Demand Growth":
        st.subheader(f"High Demand Growth ({growth_percentage}%)")
        # Apply demand growth for high demand growth scenario
        df['Demand_Model_Scenario'] = df['Demand_Model'] * (1 + (growth_percentage / 100))
        coverage_target = 0.9  # 90% coverage target for high demand
        color_scale = "Reds"  # Warmer color scale for high demand
        size_scale = 10  # Larger size for high demand
        max_stations = int(20 + (growth_percentage / 5))  # Dynamic max stations for high demand
        st.write(f"In this scenario, we model a **{growth_percentage}% growth** in demand for EV charging stations.")
    
    elif scenario == "Low Demand Growth":
        st.subheader(f"Low Demand Growth ({growth_percentage}%)")
        # Apply demand growth for low demand growth scenario
        df['Demand_Model_Scenario'] = df['Demand_Model'] * (1 + (growth_percentage / 100))
        coverage_target = 0.8  # 80% coverage target for low demand
        color_scale = "Blues"  # Cooler color scale for low demand
        size_scale = 5  # Smaller size for low demand
        max_stations = int(15 + (growth_percentage / 5))  # Dynamic max stations for low demand
        st.write(f"In this scenario, we model a **{growth_percentage}% growth** in demand for EV charging stations.")

    elif scenario == "High Traffic Volume Growth":
        st.subheader(f"High Traffic Volume Growth ({growth_percentage}%)")
        # Apply traffic volume growth for high traffic scenario
        df['Traffic_Volume_Scenario'] = df['traffc_volume'] * (1 + (growth_percentage / 100))
        beta, gamma = 0.2, 0.3  # Weights for EV vehicles and traffic volume
        df['Demand_Model_Scenario'] = (beta * df['EV Vehicles'] + gamma * df['Traffic_Volume_Scenario'])
        coverage_target = 0.9  # 90% coverage target for high traffic
        color_scale = "Oranges"  # Warmer color scale for high traffic volume
        size_scale = 10  # Larger size for high traffic volume
        max_stations = int(20 + (growth_percentage / 5))  # Dynamic max stations for high traffic volume
        st.write(f"In this scenario, we model a **{growth_percentage}% growth** in traffic volume, increasing the demand for EV charging stations.")

    elif scenario == "Low Traffic Volume Growth":
        st.subheader(f"Low Traffic Volume Growth ({growth_percentage}%)")
        # Apply traffic volume growth for low traffic scenario
        df['Traffic_Volume_Scenario'] = df['traffc_volume'] * (1 + (growth_percentage / 100))
        beta, gamma = 0.2, 0.3  # Weights for EV vehicles and traffic volume
        df['Demand_Model_Scenario'] = (beta * df['EV Vehicles'] + gamma * df['Traffic_Volume_Scenario'])
        coverage_target = 0.8  # 80% coverage target for low traffic
        color_scale = "Purples"  # Cooler color scale for low traffic volume
        size_scale = 5  # Smaller size for low traffic volume
        max_stations = int(15 + (growth_percentage / 5))  # Dynamic max stations for low traffic volume
        st.write(f"In this scenario, we model a **{growth_percentage}% growth** in traffic volume, leading to slower demand growth for EV charging stations.")

    # Filter to the top 50% demand locations for optimization
    threshold = df['Demand_Model_Scenario'].quantile(0.50)
    df_filtered = df[df['Demand_Model_Scenario'] > threshold].reset_index(drop=True)

    # Optimization Problem
    num_locations = len(df_filtered)
    problem = pulp.LpProblem("EV_Charging_Station_Optimization", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("station", range(num_locations), cat='Binary')

    # Objective function: minimize the total cost
    costs = df_filtered['Demand_Model_Scenario'].values
    problem += pulp.lpSum([costs[i] * x[i] for i in range(num_locations)])

    # Ensure the coverage of total demand (e.g., 90% or 80%)
    total_demand = df_filtered['Demand_Model_Scenario'].sum()
    problem += pulp.lpSum([df_filtered['Demand_Model_Scenario'][i] * x[i] for i in range(num_locations)]) >= coverage_target * total_demand

    # Dynamically limit the number of charging stations based on growth
    problem += pulp.lpSum([x[i] for i in range(num_locations)]) <= max_stations

    # Solve the problem
    problem.solve(pulp.PULP_CBC_CMD())

    # Extract results
    df_filtered['Station_Placement'] = [pulp.value(x[i]) for i in range(num_locations)]

    # Plot the results on a map (use only the placed stations)
    df_result = df_filtered[df_filtered['Station_Placement'] == 1]
    fig = px.scatter_mapbox(
        df_result,
        lat="latitude_population", 
        lon="longitude_population", 
        hover_name="street_name",  
        size="Demand_Model_Scenario",  # Size of the point represents demand
        color="Demand_Model_Scenario",  # Color represents demand level
        color_continuous_scale=color_scale,  # Dynamic color scale based on scenario
        size_max=size_scale,  # Dynamic size based on scenario
        zoom=10,
        height=600
    )

    # Set map style and display the map
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

    # Explanation section for Scenario
    st.subheader(f"Scenario Explanation: {scenario} ({growth_percentage}%)")
    st.write(f"The solution ensures that at least **{int(coverage_target * 100)}% of the total demand** is covered, with a maximum of {max_stations} new charging stations placed under the {scenario} scenario with {growth_percentage}% growth.")


# Section 7 - Summary/Report
if options == "Findings":
    image_path = r"C:\Users\navin\OneDrive\Desktop\Ironhack\Final Project\findingpic.png"  # Use a raw string or double backslashes
    st.image(image_path, use_column_width=True)
    st.header(" Findings and Next Steps")

    # Introduction
    #st.subheader("Introduction")
    st.write("""
    This project focused on optimizing the placement of electric vehicle (EV) charging stations in Amsterdam by analyzing high-demand areas. Using factors such as population density, traffic volume, and EV adoption rates, we developed a demand model to identify locations where new charging stations are most needed.
    """)

    # Key Findings
    st.subheader("Key Findings")
    st.write("""
    1. **High-Demand Areas Identified**: The demand model successfully identified key locations in Amsterdam where new EV charging stations are essential. These areas are characterized by high population density and heavy traffic flow, making them prime candidates for infrastructure expansion.
    2. **Insufficient Current Infrastructure**: The current 3,000 public charging stations in Amsterdam are unlikely to meet the projected EV demand by 2030, when over 1.9 million EVs are expected nationwide.
    3. **Spatial Optimization**: The clustering technique allowed us to group high-demand locations into optimal zones for new charging stations. This ensures maximum coverage while minimizing the number of stations needed.
    """)

    # Business Impact
    st.subheader("Business Impact")
    st.write("""
    - **Improved Coverage**: By strategically placing new charging stations in high-demand areas, we can significantly improve the accessibility and convenience for EV users in Amsterdam.
    - **Reduced Range Anxiety**: Optimizing station placement reduces range anxiety, encouraging more people to adopt EVs and supporting Amsterdam’s sustainability goals.
    - **Cost Efficiency**: Spatial optimization ensures that new stations are placed where they are needed most, minimizing unnecessary costs and improving infrastructure efficiency.
    """)

    # Recommendations
    st.subheader("Recommendations")
    st.write("""
    1. **Implement the New Charging Stations**: Based on the demand model, it is recommended to deploy new charging stations in the high-demand areas identified by the project.
    2. **Monitor and Adjust**: Regularly update the demand model to account for future traffic patterns, population growth, and EV adoption rates to adjust station placement as needed.
    3. **Expand to Other Cities**: The success of this model in Amsterdam can be replicated in other cities across the Netherlands, helping to build a nationwide EV infrastructure network.
    """)

    # Next Steps
    st.subheader("Next Steps")
    st.write("""
    - **Phase 1**: Begin implementation of the new charging stations in the identified areas.
    - **Phase 2**: Monitor the performance and demand for the newly placed stations and adjust the model as necessary.
    - **Phase 3**: Expand the model to other cities to help address the growing demand for EV charging stations across the Netherlands.
    """)

    # Conclusion
    st.subheader("Conclusion")
    st.write("""
    This project provides a data-driven approach to optimizing EV charging infrastructure in Amsterdam. By focusing on population density, traffic volume, and EV adoption rates, we identified high-priority locations for new stations. This will help ensure that Amsterdam is prepared to meet the growing demand for EV infrastructure, supporting the city’s transition to a more sustainable future.
    """)
