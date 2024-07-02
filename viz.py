import streamlit as st
import plotly.express as px
import pandas as pd

def app():
    st.sidebar.write('Visualization')

    if 'df' in st.session_state:
        df = st.session_state['df']
        st.write('Data Preview for Visualization')
        st.write(df)
        
        numeric_columns = list(df.select_dtypes(include=['float64', 'int64']).columns)
        non_numeric_columns = list(df.select_dtypes(exclude=['float64', 'int64']).columns)
        
        chart_select = st.sidebar.selectbox(
            label="Select the chart type",
            options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot', 'Barplot', 'Pie Chart']
        )
        
        if chart_select == 'Scatterplots':
            st.sidebar.subheader('Scatterplot Settings')
            try:
                x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
                y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
                color_value = st.sidebar.selectbox('Color by', options=numeric_columns + non_numeric_columns)
                plot = px.scatter(data_frame=df, x=x_values, y=y_values, color=color_value)
                st.plotly_chart(plot)
            except Exception as e:
                st.write("Error creating scatterplot.")
                st.write(e)
        
        if chart_select == 'Lineplots':
            st.sidebar.subheader('Lineplot Settings')
            try:
                x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
                y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
                color_value = st.sidebar.selectbox('Color by', options=numeric_columns + non_numeric_columns)
                plot = px.line(data_frame=df, x=x_values, y=y_values, color=color_value)
                st.plotly_chart(plot)
            except Exception as e:
                st.write("Error creating lineplot.")
                st.write(e)
        
        if chart_select == 'Histogram':
            st.sidebar.subheader('Histogram Settings')
            try:
                x_values = st.sidebar.selectbox('Select column', options=numeric_columns)
                color_value = st.sidebar.selectbox('Color by', options=numeric_columns + non_numeric_columns, index=len(numeric_columns + non_numeric_columns) - 1)
                plot = px.histogram(data_frame=df, x=x_values, color=color_value)
                st.plotly_chart(plot)
            except Exception as e:
                st.write("Error creating histogram.")
                st.write(e)
        
        if chart_select == 'Boxplot':
            st.sidebar.subheader('Boxplot Settings')
            try:
                y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
                x_values = st.sidebar.selectbox('X axis', options=non_numeric_columns)
                color_value = st.sidebar.selectbox('Color by', options=numeric_columns + non_numeric_columns)
                plot = px.box(data_frame=df, y=y_values, x=x_values, color=color_value)
                st.plotly_chart(plot)
            except Exception as e:
                st.write("Error creating boxplot.")
                st.write(e)
        
        if chart_select == 'Barplot':
            st.sidebar.subheader('Barplot Settings')
            try:
                x_values = st.sidebar.selectbox('X axis', options=non_numeric_columns)
                y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
                color_value = st.sidebar.selectbox('Color by', options=numeric_columns + non_numeric_columns)
                plot = px.bar(data_frame=df, x=x_values, y=y_values, color=color_value)
                st.plotly_chart(plot)
            except Exception as e:
                st.write("Error creating barplot.")
                st.write(e)
        
        if chart_select == 'Pie Chart':
            st.sidebar.subheader('Pie Chart Settings')
            try:
                names = st.sidebar.selectbox('Names', options=non_numeric_columns)
                values = st.sidebar.selectbox('Values', options=numeric_columns)
                color_value = st.sidebar.selectbox('Color by', options=numeric_columns + non_numeric_columns, index=len(numeric_columns + non_numeric_columns) - 1)
                plot = px.pie(data_frame=df, names=names, values=values, color=color_value)
                st.plotly_chart(plot)
            except Exception as e:
                st.write("Error creating pie chart.")
                st.write(e)
    else:
        st.write("No data frame available. Please upload a file in the Data Frame page.")
