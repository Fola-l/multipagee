import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np

def clean_data(df, target_column, exclude_columns):
    # Separate features and target
    X = df.drop(columns=[target_column] + exclude_columns)
    y = df[target_column]
    
    # Handle missing values and encoding
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    X = preprocessor.fit_transform(X)
    
    # Convert sparse matrix to dense numpy array
    if hasattr(X, "todense"):
        X = X.todense()
    X_dense = np.asarray(X)

    # Display cleaning steps
    cleaning_steps = (
        f"Data Cleaning Steps:\n\n1. Excluded columns: {exclude_columns}\n"
        "2. Missing values handled (mean for numeric, most frequent for categorical).\n"
        "3. Numeric features standardized.\n"
        "4. Categorical features one-hot encoded."
    )
    
    return X_dense, y, cleaning_steps, preprocessor

def app():
    st.sidebar.write('Prediction')
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.write('Data Preview for Prediction')
        st.write(df)
        
        # Column selection for prediction
        target_column = st.sidebar.selectbox('Select the column to predict', df.columns, index=0)
        
        # Column selection for exclusion
        exclude_columns = st.sidebar.multiselect('Select columns to exclude from the model', df.columns, default=[])
        
        # Data cleaning
        X, y, cleaning_steps, preprocessor = clean_data(df, target_column, exclude_columns)
        st.write('Cleaned Data Preview')
        
        try:
            st.write(pd.DataFrame(X))  # Convert to DataFrame for display
        except Exception as e:
            st.write(f"Error converting to DataFrame: {e}")
            st.write(f"Shape of X: {X.shape}")
        
        st.write(cleaning_steps)
        
        # Determine problem type (classification or regression)
        problem_type = 'classification' if y.nunique() < 10 else 'regression'
        
        # Model selection and training
        st.sidebar.subheader('Model Training')
        
        if problem_type == 'classification':
            model = RandomForestClassifier()
            param_grid = {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 10, 20, 30]
            }
        else:
            model = RandomForestRegressor()
            param_grid = {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 10, 20, 30]
            }
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            st.write(f"Best Model: {best_model}")
            
            # Hyperparameter tuning
            st.sidebar.subheader('Hyperparameter Tuning')
            n_estimators = st.sidebar.slider('Number of estimators', 10, 100, value=best_model.n_estimators)
            max_depth = st.sidebar.slider('Max depth', 1, 30, value=best_model.max_depth)
            
            best_model.set_params(n_estimators=n_estimators, max_depth=max_depth)
            best_model.fit(X_train, y_train)
            
            # Test the model
            y_pred = best_model.predict(X_test)
            
            if problem_type == 'classification':
                score = accuracy_score(y_test, y_pred)
                st.write(f"Model Accuracy: {score}")
            else:
                score = mean_squared_error(y_test, y_pred, squared=False)
                st.write(f"Model RMSE: {score}")
            
            # Input fields for prediction
            st.sidebar.subheader('Make a Prediction')
            input_data = {}
            for column in df.drop(columns=[target_column] + exclude_columns).columns:
                input_data[column] = st.sidebar.number_input(f"Input {column}", value=float(df[column].mean()))
            
            input_df = pd.DataFrame([input_data])
            transformed_input = preprocessor.transform(input_df)
            transformed_input = np.asarray(transformed_input)  # Ensure it's a numpy array
            
            if st.sidebar.button('Predict'):
                prediction = best_model.predict(transformed_input)
                
                if problem_type == 'classification':
                    st.write(f"Predicted class: {prediction[0]}")
                else:
                    st.write(f"Predicted value: {prediction[0]}")
        except Exception as e:
            st.write(f"An error occurred during model training or prediction: {e}")
    else:
        st.write("No data frame available. Please upload a file in the Data Frame page.")
