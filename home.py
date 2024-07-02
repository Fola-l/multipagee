import streamlit as st

def app():
    st.sidebar.write('Home')

    # Title and description
    st.title('Welcome to AVP: Analyze, Visualize, Predict')
    st.write('**AVP** is a comprehensive tool developed by **Noneium**, an AI solution innovation organization. AVP helps you analyze data, create visualizations, and build predictive models seamlessly.')
    
    # Introduction
    st.write("""
        ### About AVP
        AVP is designed to provide a seamless experience for data analysis, visualization, and prediction. 
        Whether you are a data scientist, analyst, or a business professional, AVP offers a suite of tools to help you gain insights from your data and make informed decisions.
    """)

    # Explanation of each page
    st.write("""
        ### Pages
        **1. Home**  
        Get an overview of what AVP offers and navigate to different sections to explore its capabilities.
        
        **2. Data Frame**  
        Upload your dataset in CSV or Excel format. This page allows you to preview your data and ensure it is properly formatted before proceeding to analysis and modeling.

        **3. Visualization**  
        Create various types of visualizations to uncover patterns and trends in your data. Choose from scatter plots, line plots, histograms, box plots, bar plots, and pie charts. Customize your charts with different settings to better understand your data.

        **4. Prediction**  
        Build and evaluate predictive models based on your dataset. Select the target variable you want to predict, clean your data, and determine the best machine learning model. Adjust hyperparameters and input values to make predictions and evaluate model performance.
    """)

    # Styling and layout
    st.markdown("""
        <style>
        .main { 
            background-color: black;
        }
        .stSidebar {
            background-color: black;
        }
        </style>
        """, unsafe_allow_html=True)

    # Footer
    st.write("""
        ---
        **Noneium**  
        Leading the way in AI-driven innovation and solutions.
    """)

if __name__ == '__main__':
    app()
