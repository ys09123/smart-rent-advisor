import streamlit as st 
import pandas as pd 
import joblib 
import os 

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Rent Advisor",
    page_icon="🏡 ",
    layout="wide"
)

# --- Caching for Performance ---
# Cache the loading of models and data to speed up the app
@st.cache_resource

def load_artifacts():
    """
    Loads the trained model, scaler, and the original dataset.
    Returns a tuple of (model, scaler, data).
    """

    # Construct paths relative to the app.py file
    # ../models/random_forest_model.pkl
    model_path = os.path.join('..', 'models', 'random_forest_model.pkl')
    scaler_path = os.path.join('..', 'models', 'scaler.pkl')
    data_path = os.path.join('..', 'data', 'raw', 'cleaned_data.csv')

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        df = pd.read_csv(data_path)

    except FileNotFoundError as e:
        st.error(f"Error loading necessary files: {e}. Please run the training script first.")
        st.stop()
    
    return model, scaler, df


# --- Load Model, Scaler, and Data ---
model, scaler, df = load_artifacts()

TRAINING_COLUMNS = ['BHK', 'City_Chennai', 'Size', 'City_Delhi', 'City_Hyderabad', 'City_Kolkata', 'City_Mumbai', 'Furnishing Status_Semi-Furnished', 'Furnishing Status_Unfurnished']

# --- UI Layout (Sidebar) ---

with st.sidebar:
    st.title("Your Renting Guide, 0% Brokerage!")
    st.header("Your Inputs")

    # 1. Salary Input
    salary = st.number_input(
        "Enter Your Monthly Salary (in Rs.)",
        min_value = 5000,
        max_value = 1000000,
        value = 50000,
        step = 1000
    )

    # 2. City Input
    city_options = sorted(df['City'].unique())
    selected_city = st.selectbox("Select Your Preferred City", options=city_options)

    # 3. BHK Input
    bhk_options = sorted(df['BHK'].unique())
    selected_bhk = st.selectbox("Select Number of Rooms (BHK)", options=bhk_options)

    # 4. Furnishing Status Input
    furnish_options = sorted(df['Furnishing Status'].unique())
    selected_furnishing = st.selectbox("Select Furnishing Status", options=furnish_options)


# --- Main Page Layout ---
st.title("Personalized Rent Advisor 🏠")
st.markdown("Find the perfect home that fits your budget and lifestyle.")
st.divider()

# --- 1. Affordability Analysis (30% of income rule) ---
st.header("Step 1: Your Affordability")
affordable_rent_30 = salary * 0.30
affordable_rent_40 = salary * 0.40

st.metric(
    "Recommended Rent Range",
    f"{affordable_rent_30:,.0f} - {affordable_rent_40:,.0f}",
    help="Based on the 30-40% rule: you should spend no than 30-40% of your gross monthly income on rent."
)

# --- 2. Personalized Rent Prediction ---
st.header("Step 2: Estimated Rent for Your Choice")

# Create a DataFrame from user inputs for prediction
def create_input_df(bhk, city, furnishing):
    #  Create a dictionary with user inputs
    user_input = {
        'BHK': bhk,
        'Size': df[(df['City'] == city) & (df['BHK'] == bhk)]['Size'].mean(),
        'City': city,
        'Furnishing Status': furnishing
    }

    input_df = pd.DataFrame([user_input])

    # One-Hot encode categorical variables
    input_encoded = pd.get_dummies(
        input_df,
        columns = ['City', 'Furnishing Status']
    )

    # Reindex to match the training columns, filling missing with 0
    input_reindexed = input_encoded.reindex(
        columns = TRAINING_COLUMNS,
        fill_value=0
    )

    return input_reindexed

# Get the processed input
input_features = create_input_df(
    selected_bhk,
    selected_city,
    selected_furnishing
)

# Scale numerical features using the loaded scaler
input_features[['BHK', 'Size']] = scaler.transform(input_features[['BHK', 'Size']])

# Predict the rent
predicted_rent = model.predict(input_features)[0]

col1, col2 = st.columns(2)
with col1:
    st.metric(
        f"Predicted Rent for a {selected_bhk} BHK in {selected_city}",
        f"{predicted_rent:,.0f}"
    )

with col2:
    if predicted_rent <= affordable_rent_40:
        st.success("This configuration seems affordable for you!")
    else:
        st.warning("This might be a stretch for your budget. Consider other options")


# ---3. Neighbourhood Recommendations ---
st.header("Step 3: Affordable Neighbourhoods")

# Filter the DataFrame based on the user's city and budget
recommendations_df = df[
    (df['City'] == selected_city) &
    (df['Rent'] >= affordable_rent_30) &
    (df['Rent'] <= affordable_rent_40) &
    (df['BHK'] == selected_bhk)
].sort_values('Rent', ascending=True).drop_duplicates(subset=['Area Locality'])

# Display Recommendations
if not recommendations_df.empty:
    st.subheader(f"Top Areas in {selected_city} Within Your Budget")
    # Display key columns in a clean table
    st.dataframe(
        recommendations_df[['Area Locality', 'Rent', 'Size', 'Furnishing Status']].head(10),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info(f"No areas found in {selected_city} for a {selected_bhk} BHK within your budget. Try adjusting your filters.")


st.divider()
st.caption("Built with ❤️ by Yash")