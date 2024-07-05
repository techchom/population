# IMPORTING ALL THE LIBRARIES
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from numerize import numerize 

# Set up the page layout and title
st.set_page_config(layout="wide", page_title="Western Balkan Population Prediction")

# Custom CSS for a more professional look
st.markdown(
    """
    <style>
       /* Custom CSS for Streamlit header */
    .st-emotion-cache-12fmjuu {
        background-color: #f48fb1; /* Pink background color for header */
        color: #ffffff; /* Text color for header */
        padding: 10px; /* Padding around header content */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); /* Shadow for header */
    }
     /* Additional custom CSS for sidebar elements */
     
     
    /* Sidebar styling */
    .st-emotion-cache-1gv3huu {
        color: #c2185b; /* Pink text color for sidebar */
        background-color: #fce4ec; /* Light pink background for sidebar */
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); /* Modified shadow */
    }
    
    /* Sidebar header and collapse button */
    .st-emotion-cache-1mi2ry5 {
        background-color: #f06292; /* Updated header background color */
        color: white; /* Text color for sidebar header */
        padding: 12px; /* Increased padding */
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
    }
    
    .stApp {
        background-color: #fce4ec; /* Light pink background color */
    }
    .css-1h0z5mz {
        color: #c2185b; /* Pink color for headings */
    }
    .css-1f7n00z {
        color: #c2185b; /* Pink color for selected text */
    }
    .css-1toay3w {
        color: #c2185b; /* Pink color for text input and selectbox */
    }
    .css-1l02z7r {
        background-color: #f8bbd0; /* Light pink background for input fields */
    }
    .css-9d5f5u {
        color: #c2185b; /* Pink color for the submit button */
    }
    .stButton>button {
        background-color: #c2185b; /* Pink color for the submit button */
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>input {
        color: #c2185b;
    }
    .stSelectbox>div>select {
        color: #c2185b;
    }   
    /* Additional custom CSS for sidebar elements */
    .css-1aumxhk, .css-1i0pu2o, .css-1v3fvcr {
        color: #c2185b; /* Pink color for sidebar headers and text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# IMPORT DATASETS
countries_pop = pd.read_csv(r'datasets/Countries_Population_final.csv')
countries_name = pd.read_csv(r'datasets/Countries_names.csv')

# Filter for Western Balkan countries
western_balkan_countries = [
    'Albania', 'Bosnia and Herzegovina', 'North Macedonia', 'Montenegro', 'Serbia', 'Kosovo'
]
countries_name = countries_name[countries_name['Country_Name'].isin(western_balkan_countries)]

# DASHBOARD TITLE
st.markdown('<h1 style="text-align: center; color: pink;">Western Balkan Population Prediction System</h1>', unsafe_allow_html=True)

# Country selection
st.sidebar.header('Select a Country and Year')
option = st.sidebar.selectbox(
    'Please Select a Country',
    sorted(countries_name['Country_Name'])
)

year_input = st.sidebar.text_input('Enter Year (1960 - 2045)', '2024')

# Display country information
country_info = {
    'Albania': 'Albania is known for its beautiful landscapes and has a growing economy.',
    'Bosnia and Herzegovina': 'Bosnia and Herzegovina is known for its rich cultural heritage and history.',
    'North Macedonia': 'North Macedonia is famous for its ancient history and diverse cultural heritage.',
    'Montenegro': 'Montenegro is known for its stunning Adriatic coastline and natural beauty.',
    'Serbia': 'Serbia has a rich history and is known for its vibrant cultural scene and historical landmarks.',
    'Kosovo': 'Kosovo is known for its complex history and recent steps towards development.',
}

st.sidebar.write(f"**Selected Country:** {option}")
st.sidebar.write(f"**Country Information:** {country_info.get(option, 'No information available.')}")
st.sidebar.write(f"**Country Population Data taken:** Data from 1960 to 2021")

# Check if the year input is a valid integer and up to 2045
if year_input.isnumeric():
    year = int(year_input)
    if year < 1960 or year > 2045:
        st.sidebar.markdown('<h4 style="color: #d32f2f;">Please enter a year between 1960 and 2045</h4>', unsafe_allow_html=True)
    else:
        X = countries_pop['Year']
        y = countries_pop[option]

        # Train Test splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        X_train = X_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)

        # Models to use
        def polynomial_regression_model(degree, year):
            poly_features = PolynomialFeatures(degree=degree)
            X_train_poly = poly_features.fit_transform(X_train)

            poly_model = LinearRegression()
            poly_model.fit(X_train_poly, Y_train)
            y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
            r2_test = r2_score(Y_test, y_test_predict)

            output = poly_model.predict(poly_features.fit_transform([[year]]))
            return output, r2_test

        def linear_regression_model(year):
            model = LinearRegression()
            model.fit(X_train, Y_train)
            y_test_predict = model.predict(X_test)
            r2_test = r2_score(Y_test, y_test_predict)

            output = model.predict([[year]])
            return output, r2_test

        def svr_model(year):
            svr_model = SVR(kernel='poly', degree=2)
            svr_model.fit(X_train, Y_train)
            y_test_predict = svr_model.predict(X_test)
            r2_test = r2_score(Y_test, y_test_predict)

            output = svr_model.predict([[year]])
            return output, r2_test

        def decision_tree_model(year):
            decision_tree = DecisionTreeRegressor()
            decision_tree.fit(X_train, Y_train)
            y_test_predict = decision_tree.predict(X_test)
            r2_test = r2_score(Y_test, y_test_predict)

            output = decision_tree.predict([[year]])
            return output, r2_test

        # Call all models
        pred_poly, r2_poly = polynomial_regression_model(2, year)
        pred_linear, r2_linear = linear_regression_model(year)
        pred_svr, r2_svr = svr_model(year)
        pred_tree, r2_tree = decision_tree_model(year)

        # OUTPUT DETAILS
        pred_pop_poly = numerize.numerize(pred_poly[0])
        pred_pop_linear = numerize.numerize(pred_linear[0])
        pred_pop_svr = numerize.numerize(pred_svr[0])
        pred_pop_tree = numerize.numerize(pred_tree[0])

        st.markdown('<h3 style="color: #c2185b;">Population Predictions for {} in {}</h3>'.format(option.upper(), year), unsafe_allow_html=True)

        st.markdown(
            """
            <table style="width:100%">
                <tr>
                    <th style="text-align:left; color: #c2185b;">Model</th>
                    <th style="text-align:left; color: #c2185b;">Predicted Population</th>
                </tr>
                <tr>
                    <td style="color: #c2185b; cursor: pointer;" onclick="document.getElementById('poly_result').style.display='block'">Polynomial Regression</td>
                    <td style="color: #c2185b;">{}</td>
                </tr>
                <tr>
                    <td style="color: #c2185b; cursor: pointer;" onclick="document.getElementById('linear_result').style.display='block'">Linear Regression</td>
                    <td style="color: #c2185b;">{}</td>
                </tr>
                <tr>
                    <td style="color: #c2185b; cursor: pointer;" onclick="document.getElementById('svr_result').style.display='block'">SVR</td>
                    <td style="color: #c2185b;">{}</td>
                </tr>
                <tr>
                    <td style="color: #c2185b; cursor: pointer;" onclick="document.getElementById('tree_result').style.display='block'">Decision Tree Regression</td>
                    <td style="color: #c2185b;">{}</td>
                </tr>
            </table>
            """.format(pred_pop_poly, pred_pop_linear, pred_pop_svr, pred_pop_tree),
            unsafe_allow_html=True
        )

        st.markdown('<h4 style="color: #c2185b;">Model Accuracies:</h4>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display:flex; justify-content: space-around;">
                <div style="text-align:center; background-color:#f8bbd0; border-radius: 10px; padding: 10px;">
                    <p>Polynomial Regression (Degree 2)</p>
                    <p>{:.2f}%</p>
                </div>
                <div style="text-align:center; background-color:#f8bbd0; border-radius: 10px; padding: 10px;">
                    <p>Linear Regression</p>
                    <p>{:.2f}%</p>
                </div>
                <div style="text-align:center; background-color:#f8bbd0; border-radius: 10px; padding: 10px;">
                    <p>SVR</p>
                    <p>{:.2f}%</p>
                </div>
                <div style="text-align:center; background-color:#f8bbd0; border-radius: 10px; padding: 10px;">
                    <p>Decision Tree Regression</p>
                    <p>{:.2f}%</p>
                </div>
            </div>
            """.format(r2_poly * 100, r2_linear * 100, r2_svr * 100, r2_tree * 100),
            unsafe_allow_html=True
        )

        # Visualization
        st.markdown('<h2 style="color: #c2185b;">Population Data and Predictions</h2>', unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=countries_pop['Year'], y=countries_pop[option], name="Historical Data",
                                line=dict(color='green', width=4)))
        fig1.add_trace(go.Scatter(x=[year], y=[pred_poly[0]], name='Polynomial Regression ' + str(year),
                                mode='markers',
                                marker_symbol='star',
                                marker=dict(size=12, color='red')))
        fig1.add_trace(go.Scatter(x=[year], y=[pred_linear[0]], name='Linear Regression ' + str(year),
                                mode='markers',
                                marker_symbol='square',
                                marker=dict(size=12, color='blue')))
        fig1.add_trace(go.Scatter(x=[year], y=[pred_svr[0]], name='SVR ' + str(year),
                                mode='markers',
                                marker_symbol='circle',
                                marker=dict(size=12, color='orange')))
        fig1.add_trace(go.Scatter(x=[year], y=[pred_tree[0]], name='Decision Tree Regression ' + str(year),
                                mode='markers',
                                marker_symbol='triangle-up',
                                marker=dict(size=12, color='purple')))
        fig1.update_layout(title='Population Data and Predictions',
                           xaxis_title='Year',
                           yaxis_title='Population',
                           legend_title='Models',
                           legend=dict(x=0, y=1.1, orientation='h'))
        st.plotly_chart(fig1)

        # Forecasting future population till 2045
        future_years = list(range(year, 2046))
        future_preds_poly = [polynomial_regression_model(2, y)[0][0] for y in future_years]
        future_preds_linear = [linear_regression_model(y)[0] for y in future_years]
        future_preds_svr = [svr_model(y)[0] for y in future_years]
        future_preds_tree = [decision_tree_model(y)[0] for y in future_years]

        future_data = pd.DataFrame({
            'Year': future_years,
            'Polynomial Regression': future_preds_poly,
            'Linear Regression': future_preds_linear,
            'SVR': future_preds_svr,
            'Decision Tree Regression': future_preds_tree
        })

        st.markdown('<h2 style="color: #c2185b;">Future Population Forecast ({} - 2045)</h2>'.format(year), unsafe_allow_html=True)
        st.write(future_data)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=future_data['Year'], y=future_data['Polynomial Regression'], name='Polynomial Regression',
                                line=dict(color='red', width=2)))
        fig2.add_trace(go.Scatter(x=future_data['Year'], y=future_data['Linear Regression'], name='Linear Regression',
                                line=dict(color='blue', width=2)))
        fig2.add_trace(go.Scatter(x=future_data['Year'], y=future_data['SVR'], name='SVR',
                                line=dict(color='orange', width=2)))
        fig2.add_trace(go.Scatter(x=future_data['Year'], y=future_data['Decision Tree Regression'], name='Decision Tree Regression',
                                line=dict(color='purple', width=2)))
        fig2.update_layout(title='Future Population Forecast ({} - 2045)'.format(year),
                           xaxis_title='Year',
                           yaxis_title='Population',
                           legend_title='Models')
        st.plotly_chart(fig2)
else:
    st.sidebar.markdown('<h4 style="color: #d32f2f;">Please enter a valid year</h4>', unsafe_allow_html=True)

# Add a footer section
st.markdown(
    """
    <footer style="text-align: center; color: #c2185b;">
        <p>Created by <strong style="color:#AA336A;">Shkurte Mustafa</strong></p>
        <p>Contact: shkurtemustafa2018@gmail.com</p>
    </footer>
    """,
    unsafe_allow_html=True
)
