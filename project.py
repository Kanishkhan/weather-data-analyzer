# weather_data_analyzer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# --------------------------
# 1. Data Loading & Cleaning
# --------------------------

# Sample Data Dictionary (if real CSV unavailable)
data = {
    'Year': list(range(2000, 2025)),
    'Temperature': [22.5, 21.8, 23.1, 22.9, 21.5, 20.8, 20.0, 19.8, 21.0, 22.3,
                    20.7, 21.1, 20.3, 19.2, 18.5, 19.0, 20.1, 21.4, 23.0, 22.7,
                    21.6, 20.9, 19.5, 20.3, 21.2],
    'Rainfall': [1200, 1300, 1250, 1400, 1100, 1000, 950, 900, 1000, 1150,
                 1050, 1300, 1250, 1100, 950, 1200, 1450, 1600, 1750, 1800,
                 1700, 1500, 1600, 1400, 1300],
    'Humidity': [65, 70, 75, 80, 60, 62, 66, 68, 74, 72,
                 70, 73, 77, 75, 65, 60, 63, 68, 70, 72,
                 74, 76, 78, 80, 82]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Handle Missing Values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

# --------------------------
# 2. Statistical Summary
# --------------------------

print("------ Statistical Summary ------")
print(df.describe())

# --------------------------
# 3. Data Visualizations
# --------------------------

# 3.1 Temperature Trend Over Years (Line Chart)
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(df['Year'], df['Temperature'], color='red', marker='o', label='Temperature (°C)')
plt.title('Temperature Trends Over Years')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend()

# 3.2 Rainfall Distribution (Bar Chart)
plt.subplot(2, 2, 2)
plt.bar(df['Year'], df['Rainfall'], color='blue', label='Rainfall (mm)')
plt.title('Yearly Rainfall Distribution')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()

# 3.3 Humidity vs Temperature (Scatter Plot)
plt.subplot(2, 2, 3)
plt.scatter(df['Temperature'], df['Humidity'], color='green', marker='x')
plt.title('Humidity vs Temperature Correlation')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.grid(True)

# 3.4 Temperature Prediction (Linear Regression)
X = df[['Year']]
y = df['Temperature']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
df['Predicted_Temp'] = model.predict(X)

# Plot actual vs predicted
plt.subplot(2, 2, 4)
plt.scatter(df['Year'], df['Temperature'], color='purple', label='Actual Temperature')
plt.plot(df['Year'], df['Predicted_Temp'], color='orange', linestyle='--', label='Predicted Trend')
plt.title('Temperature Prediction for Next Years')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()

plt.tight_layout()
plt.show()

# --------------------------
# 4. Model Evaluation
# --------------------------

mse = mean_squared_error(y, df['Predicted_Temp'])
rmse = np.sqrt(mse)

print("\n------ Model Evaluation ------")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
