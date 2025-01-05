import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv("D:\\COLLEGE DESKTOP\\BDA\\aqi_pune_2024.csv")

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# if multiple data for single day to uska average lega and then store karega
daily_data = data.resample('D').mean()

# Create lagged features for AQI prediction
for i in range(1, 6):
    daily_data[f'AQI_{i}_days_ago'] = daily_data['AQI Level'].shift(i)

# Drop karega or delete rows with empty data values
daily_data = daily_data.dropna()

# regression model use karega
features = [f'AQI_{i}_days_ago' for i in range(1, 6)]
target = 'AQI Level'

X = daily_data[features]
y = daily_data[target]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# main function to predict
def predict_aqi_for_date(day, month):
    year = 2024
    try:
        # Create a date object for the given day and month
        input_date = pd.Timestamp(year=year, month=int(month), day=int(day))

        # Temporary store karega wo calculated values till then ki dates tk
        predictions = {}
                    
        # Generate predictions from the last known date up to the input date
        current_date = daily_data.index[-1] + pd.Timedelta(days=1)  # start karega from 15th as thats the last date the csv file has

        while current_date <= input_date:
            if current_date not in daily_data.index:
                # Check for the last 5 days
                if current_date - pd.Timedelta(days=6) in daily_data.index:
                    last_day_aqi = daily_data.loc[current_date - pd.Timedelta(days=1), 'AQI Level']
                    two_days_ago_aqi = daily_data.loc[current_date - pd.Timedelta(days=2), 'AQI Level']
                    three_days_ago_aqi = daily_data.loc[current_date - pd.Timedelta(days=3), 'AQI Level']
                    four_days_ago_aqi = daily_data.loc[current_date - pd.Timedelta(days=4), 'AQI Level']
                    five_days_ago_aqi = daily_data.loc[current_date - pd.Timedelta(days=5), 'AQI Level']

                    # jo date ki calculate hogi if not equal to required date then wo wali date ke saath same function chalega loop mein
                    input_data = pd.DataFrame([[last_day_aqi, two_days_ago_aqi, 
                                                 three_days_ago_aqi, four_days_ago_aqi, 
                                                 five_days_ago_aqi]], columns=features)

                    # Use the trained model to predict AQI for the current date
                    predicted_aqi = model.predict(input_data)[0]
                    
                    # Store the prediction
                    predictions[current_date] = predicted_aqi

                    # Append the prediction to daily_data for later use
                    daily_data.loc[current_date] = [predicted_aqi] + [None]*5

            current_date += pd.Timedelta(days=1)

        # jab value input denge to ye wala function output dega
        if input_date in predictions:
            print(f'Predicted AQI for {input_date.strftime("%d-%m-%Y")}: {predictions[input_date]}')
        else:
            print(f"No prediction available for {input_date.strftime('%d-%m-%Y')}.")
            return

        # Save the updated daily data with new predictions to CSV
        # daily_data.to_csv("D:\\COLLEGE DESKTOP\\BDA\\aqi_pune_2024_with_predictions.csv")

    except Exception as e:
        print(f"Error: {e}")


# Ask the user for the date (day and month) in DD-MM format for which they want the AQI prediction
user_input = input("Enter the date (DD-MM) for which you want the AQI prediction in 2024: ")

# Split the input into day and month
day, month = user_input.split('-')

# call karega main function and predict karega
predict_aqi_for_date(day, month)

# Optional: Save the model for future use
joblib.dump(model, 'aqi_prediction_model.pkl')
