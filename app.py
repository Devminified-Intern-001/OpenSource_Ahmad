from flask import Flask, jsonify, request
import pandas as pand
import numpy as np
import xgboost as xgb
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define the default model name
default_model_name = 'Central 3.json'

# Load the trained XGBoost model
reg = xgb.XGBRegressor()
reg.load_model(default_model_name)

# Define function to create features including lagged variables
def create_features_with_lags(df):
    """
    Create time series features based on time series index and include lagged variables.
    """
    df = df.copy()
    df['dayofyear'] = df.index.dayofyear
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # Add lagged features for the future dates
    df['lag1'] = df['dayofyear'].shift(periods=364)
    df['lag2'] = df['dayofyear'].shift(periods=728)
    df['lag3'] = df['dayofyear'].shift(periods=1092)
    
    return df

# Define prediction function
def predict_load_shedding(model_name, from_date, to_date):
    # Load the specified model
    reg.load_model(model_name)
    
    future_dates = pand.date_range(from_date, to_date, freq='D')
    future_df = pand.DataFrame(index=future_dates)
    future_df['isFuture'] = True
    future_df = create_features_with_lags(future_df)  # Upandate to use create_features_with_lags
    future_predictions = reg.predict(future_df.drop(columns=['isFuture']))  # Drop 'isFuture' column
    
    # Convert float32 predictions to regular Python floats
    future_predictions = future_predictions.astype(float)
    
    # Round predictions to 2 decimal places
    future_predictions_rounded = [round(prediction, 2) for prediction in future_predictions]
    
    # Create a dictionary to store rounded predictions for each date
    predictions_dict = {date.strftime('%Y-%m-%d'): rounded_prediction for date, rounded_prediction in zip(future_dates, future_predictions_rounded)}
    
    return predictions_dict

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request sent by React component
    data = request.get_json()
    selected_main_region = data.get('selectedMainRegion')
    selected_com_region = data.get('selectedComRegion')
    selected_mbu = data.get('selectedMbu')
    from_date = data.get('fromDate')
    to_date = data.get('toDate')
    unit_price = float(data.get('unitPrice'))  # Convert unitPrice to float

    # Construct model name based on selected region
    if selected_main_region:
        model_name = f"{selected_main_region}.json"
    elif selected_com_region:
        model_name = f"{selected_com_region}.json"
    elif selected_mbu:
        model_name = f"{selected_mbu}.json"
    else:
        model_name = default_model_name

    # Make predictions
    predictions = predict_load_shedding(model_name, from_date, to_date)

     # Perform bill calculations
    bill_results = {}
    for date, prediction in predictions.items():
        bill_result = ((24 - prediction) * 8) * unit_price
        bill_results[date] = round(bill_result, 2)

    # Prepare response
    response = {
        "predictions": predictions,
        "bill_results": bill_results
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)