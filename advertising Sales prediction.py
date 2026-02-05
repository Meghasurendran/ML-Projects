# Advertising Sales Prediction
# Multiple Linear Regression
# ============================================

# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

# Load Dataset

df = pd.read_csv("advertising.csv")

print(df.head())

print("\nDataset Info\n")
print(df.info())

print("\nMissing Values\n")
print(df.isna().sum())

print("\nStatistical Summary\n")
print(df.describe())

# Correlation Analysis

plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Data Visualization

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
sns.regplot(x=df['TV'], y=df['Sales'], color='r')
plt.title("TV vs Sales")

plt.subplot(1,3,2)
sns.regplot(x=df['Radio'], y=df['Sales'], color='b')
plt.title("Radio vs Sales")

plt.subplot(1,3,3)
sns.regplot(x=df['Newspaper'], y=df['Sales'], color='g')
plt.title("Newspaper vs Sales")

plt.tight_layout()
plt.show()

# Split Features & Target

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nFeatures (X)\n")
print(X)

print("\nTarget (y)\n")
print(y)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Train Model

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Actual vs Predicted

result_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual vs Predicted Values\n")
print(result_df)

# Visualization: Actual vs Predicted

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()


# Model Parameters

print("\nIntercept (Constant):", model.intercept_)

print("\nCoefficients (Slopes):")

for col, coef in zip(X.columns, model.coef_):
    print(col, ":", coef)

# Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance\n")

print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("R2 Score:", r2)

new_data = pd.DataFrame([[130, 59, 45]], columns=X.columns)
new_prediction = model.predict(new_data)
print("\nPrediction for New Data (TV=130, Radio=59, Newspaper=45):")
print("Predicted Sales:", new_prediction[0])