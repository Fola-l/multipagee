import streamlit as st
import pandas as pd

# Function to read the uploaded file and cache the result
@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        return pd.read_excel(file)

def app():
    st.write('DataFrame')

    # Check if a data frame is already in session state
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.write('Data Preview')
        st.write(df)
    else:
        # File uploader on the main page
        uploaded_file = st.file_uploader(label="Upload your CSV or Excel File", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state['df'] = df  # Store the data frame in session state
                st.write('Data Preview')
                st.write(df)
            else:
                st.write("There was an error processing the file.")
        else:
            st.write("Please upload a file to the application.")
