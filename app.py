import sqlite3
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import logging

# Load scaler and model
std_scaler = joblib.load('./models/std_scaler.lb')
kmeans_model = joblib.load('./models/kmeans_model.lb')
df = pd.read_csv("./models/filter_crops.csv")

app = Flask(__name__)

# Set up SQLite database
DATABASE = 'farmer_data.db'

def init_db():
    """ Initialize the database and create the table if it doesn't exist """
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS FarmerData (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            N INTEGER NOT NULL,
            P INTEGER NOT NULL,
            K INTEGER NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            PH REAL NOT NULL,
            rainfall REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_data(N, P, K, temperature, humidity, PH, rainfall):
    """ Insert farmer data into the SQLite database """
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO FarmerData (N, P, K, temperature, humidity, PH, rainfall)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (N, P, K, temperature, humidity, PH, rainfall))
    conn.commit()
    conn.close()

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from form and validate inputs
            N = int(request.form['N'])
            if not (0 <= N <= 100):
                return "Invalid value for Nitrogen content."
                
            PH = float(request.form['PH'])
            if not (0 <= PH <= 14):
                return "Invalid value for PH level."

            P = int(request.form['P'])
            K = int(request.form['K'])
            humidity = float(request.form['humidity'])
            rainfall = float(request.form['rainfall'])
            temperature = float(request.form['temperature'])

            # Prepare the data for model
            UNSEEN_DATA = [[N, P, K, temperature, humidity, PH, rainfall]]
            transformed_data = std_scaler.transform(UNSEEN_DATA)

            # Make prediction
            cluster = kmeans_model.predict(transformed_data)[0]
            suggestion_crops = list(df[df['cluster_no'] == cluster]['label'].unique())

            # Store data into the SQLite database
            insert_data(N, P, K, temperature, humidity, PH, rainfall)
            logging.info("Data inserted into SQLite")

            # Render the output page with the suggested crops
            return render_template('output.html', crops=suggestion_crops)

        except Exception as e:
            # Handle any exceptions and provide error message
            logging.error(f"An error occurred: {e}")
            return f"An error occurred during prediction: {e}"

if __name__ == "__main__":
    # Initialize the database
    init_db()

    # Run the Flask app with debug mode off for production
    app.run(debug=False)
