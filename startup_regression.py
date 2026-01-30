# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

# Load Dataset
url = "https://raw.githubusercontent.com/arib168/data/main/50_Startups.csv"
df = pd.read_csv(url)

print("First 5 Rows:")
print(df.head())

# Basic Data Exploration
print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isna().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# Exploratory Data Analysis (EDA)

# State Distribution (Pie Chart)
state_count = df['State'].value_counts()

plt.figure()
plt.pie(state_count, labels=state_count.index, autopct='%1.1f%%')
plt.title("Startup Distribution by State")
plt.show()


# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    df.select_dtypes(include=[np.number]).corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()


# Regression Plots
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.regplot(x=df['R&D Spend'], y=df['Profit'])
plt.title("R&D Spend vs Profit")

plt.subplot(1, 3, 2)
sns.regplot(x=df['Administration'], y=df['Profit'])
plt.title("Administration vs Profit")

plt.subplot(1, 3, 3)
sns.regplot(x=df['Marketing Spend'], y=df['Profit'])
plt.title("Marketing Spend vs Profit")

plt.tight_layout()
plt.show()


# Feature & Target Split
X = df.drop(columns=['Profit'])
y = df['Profit']


# Categorical Encoding
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['State']),
    remainder='passthrough'
)

X = preprocessor.fit_transform(X)


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

print("\nTrain-Test Split Completed!")


# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Training Completed!")


# Prediction
y_pred = model.predict(X_test)


# Actual vs Predicted Comparison
results = pd.DataFrame({
    "Actual Profit": y_test,
    "Predicted Profit": y_pred
})

print("\nActual vs Predicted Values:")
print(results.head())


# Scatter Plot: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    '--r'
)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.show()


# Model Parameters
print("\nModel Intercept:", model.intercept_)

print("\nModel Coefficients:")
print(model.coef_)


# Performance Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("R2 Score:", r2)
