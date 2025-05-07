import streamlit as st
import pandas as pd
import numpy as np

st.title('Basketball Points App')
st.write("This app shows the player's information from the 1996 season to the 2022 season.")

# Read CSV file containing player data
df = pd.read_csv(r"C:\Users\kevin\OneDrive\Documents\Rankel-Python-Project\basic-streamlit-app\all_seasons.csv")

# Display the entire dataframe for an initial overview
st.dataframe(df)

# Convert heights from centimeters to inches, rounding to nearest whole number
df['player_height'] = round(df['player_height'].astype(float) / 2.54)

# Convert weights from kilograms to pounds, rounding to nearest whole number
df['player_weight'] = round(df['player_weight'].astype(float) * 2.20462)

# Extract the starting year of the season from 'season' column and convert it to integer
df['season_start'] = df['season'].astype(str).str[:4].astype(int)

# Determine the minimum and maximum season years for slider range
min_season = df['season_start'].min()
max_season = df['season_start'].max()

# Create a slider for selecting the minimum season, affecting the data filtered below
season = st.slider('Choose a minimum season', min_season, max_season, min_season)

# Filter the dataframe to include only data from the selected season onwards
filtered_df = df[df['season_start'] >= season]

# Get a sorted list of unique team abbreviations, ensuring there are no missing values
teams = sorted(df['team_abbreviation'].dropna().unique())

# Dropdown for selecting a team, with "All Teams" as the first option
selected_team = st.selectbox('Select a team', ['All Teams'] + teams)

# Further filter the dataframe to show only the selected team, unless "All Teams" is selected
if selected_team != 'All Teams':
    filtered_df = filtered_df[filtered_df['team_abbreviation'] == selected_team]

# Output the current filtering criteria
st.write(f'Showing data for {selected_team} in seasons {season} and later: ')

# Check if the resulting dataframe after filtering is empty
if filtered_df.empty:
    st.write("No data available for the selected criteria.")
else:
    # Display the filtered dataframe
    st.dataframe(filtered_df)

    # Check if the 'pts' column exists in the filtered dataframe
    if 'pts' in filtered_df.columns:
        # Coerce non-numeric values in 'pts' to NaN, ensuring the data in the column is numeric
        filtered_df['pts'] = pd.to_numeric(filtered_df['pts'], errors='coerce')

        # Visualization: Histogram of total points scored up to 40 points
        st.write("Distribution of Total Points Scored by Players:")

        # Create bins for every integer point total from 0 to 40
        points_bins = np.arange(0, 41, 1)

        # Use numpy's histogram to count how many times each score appears within the range, bins define the score range
        hist_values, bin_edges = np.histogram(filtered_df['pts'].dropna(), bins=points_bins)
        st.bar_chart(hist_values)  # Display histogram as a bar chart

        # Visualization: Line chart for average points scored by players at each age
        avg_points_by_age = filtered_df.groupby('age')['pts'].mean()
        st.write("Average Points by Age:")
        st.line_chart(avg_points_by_age)  # Display line chart of average points by age
    else:
        # Inform the user if the 'pts' column is missing
        st.write("The dataset does not have a 'pts' column to visualize the scoring data.")