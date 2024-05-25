import os
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='/static')
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'  # Set secret key for session

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_data(df):
    try:
        print("Before preprocessing:", df.shape)
        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')  # Update date format
        
        # Drop rows with missing values in specific columns
        df = df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'])
        
        print("After preprocessing:", df.shape)
        df = df.sort_values(by='Date')
        return df
    except Exception as e:
        print("Error during preprocessing:", e)
        return None

def train_model(X_train, y_train):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),  # Add an Input layer specifying the input shape
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, verbose=1)  # Changed verbose to 1 for printing training progress
    print("Model training successful.")
    return model

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Print the content of the uploaded CSV file
        df_uploaded = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("Uploaded CSV file content:")
        print(df_uploaded.head())
        
        return redirect(url_for('predict', filename=filename))

@app.route('/predict/<filename>', methods=['GET', 'POST'])
def predict(filename):
    if request.method == 'POST':
        try:
            # Handle form submission and perform prediction
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            df = preprocess_data(df)
            
            if df is not None and not df.empty:
                scaler = MinMaxScaler()
                X = scaler.fit_transform(df[['Open', 'High', 'Low', 'Adj Close', 'Volume']])
                y = df['Close'].values
                X = X.reshape((X.shape[0], X.shape[1], 1))
                
                # Train the model (assuming train_model function is defined)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = train_model(X_train, y_train)
                
                # Perform prediction based on the input date
                input_date = pd.to_datetime(request.form['date'])
                input_data = df[df['Date'] == input_date][['Open', 'High', 'Low', 'Adj Close', 'Volume']]
                input_data_scaled = scaler.transform(input_data)
                input_data_scaled = input_data_scaled.reshape((input_data_scaled.shape[0], input_data_scaled.shape[1], 1))
                prediction = model.predict(input_data_scaled)[0][0]
                
                # Retrieve the previous value for comparison
                previous_value = df[df['Date'] < input_date]['Close'].values[-1]
                
                # Store prediction and previous value in session
                session['prediction'] = float(prediction)
                session['previous_value'] = float(previous_value)
                
                return redirect(url_for('result'))
            else:
                return "Error: Empty dataset after preprocessing."
        except Exception as e:
            return "Error: " + str(e)
    else:
        # Display form for entering the date
        return render_template('predict.html', filename=filename)

@app.route('/result')
def result():
    # Retrieve prediction and previous value from session
    prediction = session.pop('prediction', None)
    previous_value = session.pop('previous_value', None)
    
    # Convert the prediction and previous value to float
    prediction = float(prediction) if prediction is not None else None
    previous_value = float(previous_value) if previous_value is not None else None
    
    if prediction is not None and previous_value is not None:
        # Calculate the change
        change = prediction - previous_value
        
        # Calculate the percentage change
        percentage_change = (change / previous_value) * 10
        
        # Format the output
        formatted_output = "{:,.2f}".format(prediction) + "-" + "{:,.2f}".format(abs(change)) + " (" + "{:.2f}".format(percentage_change) + "%)"
        
        return render_template('result.html', prediction=previous_value, previous_value=prediction, percentage_change=percentage_change)
    else:
        return "Error: Prediction or previous value not available."

if __name__ == '__main__':
    app.run(debug=True)