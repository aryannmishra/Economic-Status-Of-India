import streamlit as st 
import pickle
from streamlit_option_menu import option_menu
import numpy as np
import time
import pandas as pd
import plotly.express as px
from datetime import date
from sklearn.preprocessing import LabelEncoder
import bcrypt
import sqlite3
import geopandas as gpd
import matplotlib.pyplot as plt

# Implementing the design
st.set_page_config(
    page_title="Economy Prediction & Analysis",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/ArunKumar76',
        'Report a bug': "https://www.linkedin.com/in/arun-kumar-598072253/",
        'About': "# Made by Arun Kumar!"
    }
)

# Loading the saved Random Forest model
@st.cache_resource(show_spinner=True)
def get_model(model_name):
    model = pickle.load(open(model_name, 'rb'))
    return model

RFmodel = get_model('trained_model2.pkl')

# Replace this with the list of all states used during model training
states = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", 
          "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", 
          "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", 
          "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Delhi", "Chandigarh", "Puducherry", "Jammu and Kashmir", ""]

# Initialize the label encoder and fit on the list of states
label_encoder = LabelEncoder()
label_encoder.fit(states)

# Save the fitted label encoder to a file
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

@st.cache_resource(show_spinner=True)
def get_label_encoder():
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    return label_encoder

label_encoder = get_label_encoder()

# ------------------LOGIN---------------------------
st.title("ECONOMIC ANALYSIS AND PREDICTION OF INDIAN STATES")

# Create User Table
def create_user_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                 username TEXT NOT NULL, 
                 password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def signup():
    if 'username' in st.session_state and len(st.session_state['username']) > 0:
        st.success(f"Hey {st.session_state['username'].capitalize()}, you are logged in already")
        return

    st.write("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')

    if st.button("Signup", type='primary', use_container_width=True):
        if password == confirm_password:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            conn.close()
            st.success("You have successfully created an account!")
            st.info("Please login to proceed.")
        else:
            st.warning("Passwords do not match.")

def login():
    if 'username' in st.session_state and len(st.session_state['username']) > 0:
        st.write("### Hey", ':blue[', st.session_state.username.capitalize() + ' 👋', ']')
        return

    st.write("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login", type='primary', use_container_width=True):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        result = c.fetchone()
        conn.close()
        if result:
            hashed_password = result[2]
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                st.session_state['username'] = username
                st.success("You have successfully logged in!")
            else:
                st.warning("Incorrect Password")
                st.stop()
        else:
            st.markdown("## :red[Username not found, Please Sign Up!]")
            st.stop()

def logout():
    if len(st.session_state['username']) <= 0:
        st.warning("You have not logged in yet!")
        return

    st.session_state['username'] = ''
    st.success("You have successfully logged out!")

def main():
    create_user_table()
    menu = ["Login", "Signup", "Logout"]
    choice = st.sidebar.selectbox("Select an option", menu)

    if choice == "Signup":
        signup()
    elif choice == "Login":
        login()
    elif choice == "Logout":
        logout()

if __name__ == '__main__':
    main()

if 'username' in st.session_state and len(st.session_state['username']) > 0:
    st.sidebar.title('Hi :blue[' + st.session_state.username.capitalize() + '!]')
else:
    st.stop()

with st.sidebar:
    selected = option_menu(
        menu_title="Main menu",
        options=["Prediction", "Analytics", "Recommendation", "Help"],
    )
st.header(f"{selected}")

if selected == 'Prediction':
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        latest_iteration.text(f'App is getting ready... {i+1}%')
        bar.progress(i + 1)
        time.sleep(0.01)
    
    tab1, = st.tabs(["India"])

    with tab1:
        st.markdown('''
        :orange[Disclaimer]
        
        Feel free to change the values to predict the Economic Status of State!''')

        # INPUT VALUES FROM THE USER
        att_stat = st.text_input('Enter the name any State OR Union Trritory:', value='Uttar Pradesh')
        if att_stat not in states:
            st.error(f"Invalid State: `{att_stat}`. Please enter a valid state name.")
            st.stop()
        # Encode state input
        encoded_state = label_encoder.transform([att_stat])  # Use the loaded encoder
        user_input = np.array(encoded_state).reshape(1, -1)  # Correct format for prediction
        
        if st.button('__**Predict Economic_Status **__', use_container_width=True, type='primary'):
            prediction = RFmodel.predict(user_input)

            with st.spinner('Prediction is on the way...'):
                time.sleep(1)
            

            with st.container():
                st.balloons()
                st.snow()

                # Iterate over predictions and display each on a new line
                for i, pred in enumerate(prediction):
                    st.header(f'Per Capita Income: `{pred[0]}`')
                    st.header(f'Mortality Rate: `{pred[1]}`')
                    st.header(f'GDP: `{pred[2]}`')
                    st.header(f'Birth Rate: `{pred[3]}`')
                    st.header(f'Life Expectancy: `{pred[4]}`')
                    st.header(f'Population: `{pred[5]}`')
                    st.header(f'Literacy Rate: `{pred[6]}`')
                    st.header(f'Economic_Score: `{pred[7]}`')
                st.success(f'R2 Score of the _Gradient Boosting Regressor_ is: __{0.84}__')
                st.info('Generally R2 score __>0.7__ is considered as good', icon="ℹ️")
        
        #Showing The Heatmap
        india_map = gpd.read_file("Indian_map\Indian_States.shp")
        # Ensure the state names match between your dataset and the shapefile
        india_map = india_map.rename(columns={'st_nm': 'State'})

        # Select numeric columns for filling missing values
        numeric_columns = india_map.select_dtypes(include=['number']).columns
        india_map[numeric_columns] = india_map[numeric_columns].fillna(india_map[numeric_columns].mean())

        #Load filtered data
        df = pd.read_csv('filtered_data.csv')
        india_map['State'] = india_map['State'].str.strip()
        df['State'] = df['State'].str.replace(r'\s+', ' ', regex=True).str.strip()
        # Merge shapefile with your economic data
        india_map = india_map.merge(df,how='left', on='State')

        # Select numeric columns for filling missing values
        numeric_columns = india_map.select_dtypes(include=['number']).columns
        india_map[numeric_columns] = india_map[numeric_columns].fillna(india_map[numeric_columns].mean())
        
        # Create a selectbox for Economic Indicator
        valid_indicators = [col for col in df.columns if col != 'State']  # List of indicators in your dataset
        eco_indi = st.selectbox('Select Economic Indicator for Heatmap:', valid_indicators)
    
        if st.button('__**Show Heatmap**__', use_container_width=True, type='primary'):
            st.header(f'Heatmap is Showing for `{eco_indi}`')
            # Plotting
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            india_map.plot(column=eco_indi, cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
            for idx, row in india_map.iterrows():
                #value = row[eco_indi]
                ax.text(x=row['geometry'].centroid.x,
                        y=row['geometry'].centroid.y,
                        s=row['State'],
                        #s=f'{value:.2f}',  # Format the value with 2 decimal places
                        fontsize=8,
                        ha='center',
                        color='black')
            ax.set_title(f'{eco_indi} Prediction by State')
            st.pyplot(fig)
            