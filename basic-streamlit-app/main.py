import streamlit as st
import pandas as pd
st.title('Basketball Points App')
st.write("This app shows the distribution for a player's past five games to compare them. It also takes the average for those five games and has a sliding scale to show those who averaged more than a certain number.")
df=pd.DataFrame({
    'Player': ['Lebron James', 'Shai Gilgeous-Alexander', 'Nikola Jokic', 'Luka Doncic', 'Jayson Tatum'],
    'Game 1': [42, 50, 28, 14, 17],
    'Game 2': [26, 34, 38, 27, 22],
    'Game 3': [33, 29, 27, 45, 35],
    'Game 4': [24, 52, 28, 16, 27],
    'Game 5': [31, 35, 28, 30, 16]})
df['Average']=df.iloc[:,1:].mean(axis=1).round(2)
st.dataframe(df)

average=st.slider('Choose a minimum average',
min_value=df['Average'].min(),
max_value=df['Average'].max())
st.dataframe(df[df['Average']>=average])