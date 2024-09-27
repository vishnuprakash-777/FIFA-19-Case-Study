import streamlit as st
import pandas as pd
from PIL import Image

# Page configuration
st.set_page_config(page_title='FIFA 2019 Player Attributes', page_icon='⚽', layout='wide')

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .header {
        background-color: #00274d;
        color: white;
        padding: 1rem;
        text-align: center;
        border-radius: 10px;
    }
    .footer {
        background-color: #00274d;
        color: white;
        padding: 0.5rem;
        text-align: center;
        border-radius: 10px;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header with Logo
# header_image = Image.open("header_logo.png")  # Uncomment and replace with your logo file if needed
# st.image(header_image, width=100)
st.markdown("<div class='header'><h1>FIFA 2019 Player Attributes Analysis</h1></div>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose an option", ["Overview", "Dataset"])

if option == "Overview":
    st.write("""
        ### Explore and analyze the attributes of FIFA 2019 players.

        This app allows you to explore and analyze various attributes of FIFA 2019 players. The dataset includes comprehensive details about each player, such as:

        - **Personal Information**: Age, Nationality, Club, Preferred Foot, International Reputation, Height, Weight, Jersey Number, Joined Date, Loaned From, Contract Validity.
        - **Skills and Attributes**: Overall, Potential, Value, Wage, Work Rate, Position.
        - **Detailed Skill Ratings**: Crossing, Finishing, Heading Accuracy, Short Passing, Volleys, Dribbling, Curve, FK Accuracy, Long Passing, Ball Control, Acceleration, Sprint Speed, Agility, Reactions, Balance, Shot Power, Jumping, Stamina, Strength, Long Shots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, Standing Tackle, Sliding Tackle.
        - **Goalkeeper Attributes**: GK Diving, GK Handling, GK Kicking, GK Positioning, GK Reflexes.
        - **Player Position Ratings**: LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB.
        - **Release Clause**: The amount a club must pay to release a player from their contract.

        Scroll down to explore the dataset.
    """)

elif option == "Dataset":
    # Load dataset
    @st.cache_data
    def load_data():
        try:
            return pd.read_csv('kl.csv', encoding='ISO-8859-1', low_memory=False)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

    data = load_data()

    if not data.empty:
        st.header('FIFA 2019 Player Dataset')
        st.write(data)
    else:
        st.write("No data available to display.")

#elif option == "Visualizations":
    #st.write("Visualizations section coming soon!")
    # Here you can add various visualizations such as histograms, scatter plots, etc.

# Footer
st.markdown("<div class='footer'><p>&copy; 2019 FIFA Analysis.</p></div>", unsafe_allow_html=True)
