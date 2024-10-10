EV Charging Station Optimization Project
Project Overview:

This project focuses on optimizing the placement of electric vehicle (EV) charging stations in Amsterdam, using data-driven techniques to identify high-demand areas based on population density, traffic volume, and the number of electric vehicles (EVs). The project incorporates web scraping, external APIs, and custom data calculations to deliver a comprehensive solution for charging infrastructure planning.

Key Features:

Data Collection:
Charging Stations: Web scraped from Electromaps, a site that provides real-time information on charging stations across various locations.
Population and EV Vehicle Data: Sourced from Gemeente Amsterdam, the official government portal for Amsterdam's population and vehicle data.
Geolocation Data (Lat/Long): Collected using the Google API to retrieve latitude and longitude coordinates for various streets in Amsterdam.
Exploratory Data Analysis (EDA):

Visualizes the current distribution of EV charging stations in Amsterdam.
Provides insights into population density and traffic volume, helping to understand how these factors contribute to the overall demand for EV charging infrastructure.

Demand Modeling:
Uses a demand model that takes into account population density, number of EV vehicles (hypothetically calculated for each street), and traffic volume to calculate high-demand areas for EV charging stations.
The demand model is flexible and can be adjusted based on growth projections.
Scenario Analysis:

Allows users to explore different scenarios:
Moderate Demand Growth (40%): Projects a 40% increase in demand for charging stations.
Low Demand Growth (10%): Projects a 10% increase in demand for charging stations.
Shift in Traffic Patterns: Adjusts traffic volume in some areas, increasing demand in high-traffic areas and reducing it in low-traffic areas.
Each scenario shows how the demand for EV charging stations changes based on different projections.
Optimization Model:

Identifies optimal locations for new charging stations using clustering techniques and spatial optimization.
Optimized locations are displayed on an interactive map, providing a clear visualization of where charging stations should be added.
Interactive Visualization:

Users can interact with the app by selecting different scenarios to see how charging station needs evolve .

Includes dynamic maps to visualize:
Current charging stations.
High-demand areas for new charging stations based on the demand model.
Optimized charging station locations based on spatial clustering.

Data Collection Methods:

Charging Stations (Web Scraping from Electromaps):

The charging station data was collected by web scraping the Electromaps website, which lists all public charging stations in Amsterdam.
The scraped data includes information such as station name, address, connector type, and capacity.
Population and EV Vehicle Data (Gemeente Amsterdam):

Population and vehicle data were obtained from the Gemeente Amsterdam portal, which provides open datasets for public use. This data includes population per street and EV ownership statistics.
Geolocation Data (Google API):

Latitude and longitude coordinates for each street in Amsterdam were retrieved using the Google Maps API. This allowed for precise geospatial mapping of the demand model and charging stations.
Hypothetical Calculations:

Population density was calculated for each street based on the available population data and geographic area of each neighborhood.
EV vehicle counts were hypothetically calculated for each street using growth rates and population-based projections to estimate the future number of EVs.
Installation & Setup:
Install Python Dependencies: Make sure you have the following libraries installed. You can install them using the requirements.txt or manually as follows:

bash
Copy code
pip install streamlit pandas plotly folium scikit-learn
Run the Streamlit App: To run the project, navigate to the project folder and run the following command:

bash
Copy code
streamlit run main.py
This will launch the Streamlit web app in your default web browser.

Dataset Structure:

Charging_Station_Cleaned.csv: Contains details of existing charging stations.
normalized_population_traffic_with_demand.csv: Contains the demand model and traffic data.
cluster_centroids.csv: Contains optimized locations for new charging stations.
Make sure that all datasets are in the correct file paths, or modify the paths in the code where the datasets are loaded.

Project Structure:
bash
Copy code

📁 EV_Charging_Station_Optimization
│
├── Projct.py                   # Main Streamlit application file
├── Charging_Station_Cleaned.csv     # Charging stations data
├── normalized_population_traffic_with_demand.csv  # Demand model data
├── cluster_centroids.csv      # Optimized locations data
├── images                     # Folder for images used in the Streamlit app
│   └── AdobeStock.jpg
│   └── findingpic.png
│   └── technical.webp
├── requirements.txt           # List of required Python libraries
└── README.md                  # Project description and instructions


How to Use the Application:

Explore different growth scenarios:
High Demand Growth
Low Demand Growth
High Traffic Volume Growth
Low Traffic Volume Growth

Explore the Interactive Maps:

View the current distribution of charging stations in Amsterdam.
Visualize high-demand areas for new charging stations.
See the optimized locations for adding new stations based on the selected scenario.

Scenario Analysis:

Understand how different growth projections or changes in traffic patterns will impact the need for EV charging stations.
Key Considerations:
Data Collection: Web scraping, Google API requests, and manual data calculations were used. Ensure that the sources are reliable and regularly updated for future iterations of the model.

Hypothetical Data: Some data points, like EV vehicles per street, were hypothetically calculated based on population projections. These figures should be verified or updated as more accurate data becomes available.

Future Enhancements:
Real-Time Data Integration: Integrate real-time traffic and population data to continuously update the demand model.
Budget Constraints: Introduce optimization that factors in budget limitations and prioritizes areas with the highest return on investment for charging infrastructure.
Public Input: Include a feedback mechanism where users can provide input or preferences for where charging stations should be placed.
Authors:
Prerna Shrivastava

This project was developed as part of the EV Charging Station Optimization initiative for the city of Amsterdam, aiming to create a sustainable infrastructure for electric vehicles.

