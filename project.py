import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset matching the image output
data = {
    'Year': list(range(2000, 2025)),
    'Temperature': [21, 20, 22, 23, 20, 19, 23, 21, 19, 20,
                    16, 19, 18, 21, 19, 16.2, 18.5, 20.7, 22.9, 19.6,
                    20.1, 17.1, 19.4, 16.3, 18.9],
    'Rainfall': [1500, 800, 1600, 1200, 1350, 1100, 1400, 1150, 900, 1580,
                 1300, 1100, 1750, 1050, 1250, 1350, 1650, 1750, 1900, 1850,
                 1750, 1350, 1500, 1250, 1100],
    'Humidity': [62, 70, 55, 72, 59, 66, 83, 54, 77, 67,
                 48, 70, 75, 63, 79, 85, 60, 71, 68, 80,
                 74, 85, 55, 49, 70]
}

# Create DataFrame
df = pd.DataFrame(data)

# Fill missing values if any
df.fillna(df.mean(numeric_only=True), inplace=True)

# Prepare for prediction
X = df[['Year']]
y = df['Temperature']

# Linear Regression model
model = LinearRegression()
model.fit(X, y)
df['Predicted_Temp'] = model.predict(X)

# ----------------- Plotting -----------------

plt.figure(figsize=(12, 8))

# 1. Temperature Trends Over Years
plt.subplot(2, 2, 1)
plt.plot(df['Year'], df['Temperature'], color='red', marker='o', label='Temperature (°C)')
plt.title('Temperature Trends Over Years')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend()

# 2. Yearly Rainfall Distribution
plt.subplot(2, 2, 2)
plt.bar(df['Year'], df['Rainfall'], color='blue', label='Rainfall (mm)')
plt.title('Yearly Rainfall Distribution')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()

# 3. Humidity vs Temperature Correlation
plt.subplot(2, 2, 3)
plt.scatter(df['Temperature'], df['Humidity'], color='green', marker='x')
plt.title('Humidity vs Temperature Correlation')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.grid(True)

# 4. Temperature Prediction for Next Years
plt.subplot(2, 2, 4)
plt.scatter(df['Year'], df['Temperature'], color='purple', label='Actual Temperature', marker='x')
plt.plot(df['Year'], df['Predicted_Temp'], color='orange', linestyle='--', label='Predicted Trend')
plt.title('Temperature Prediction for Next Years')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()

# Layout adjustment
plt.tight_layout()
plt.show()

# ----------------- Model Evaluation -----------------
mse = mean_squared_error(y, df['Predicted_Temp'])
rmse = np.sqrt(mse)

print("------ Model Evaluation ------")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

