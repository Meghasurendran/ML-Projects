# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

# Load Dataset
df = pd.read_csv("diabetes.csv")

print(df.head())
print(df.columns)
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.dtypes)


# Exploratory Data Analysis

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# Target Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes Outcome Distribution")
plt.show()


# Pairplot
sns.pairplot(df, hue="Outcome")
plt.show()


# Boxplot (Before Outlier Handling)
plt.figure(figsize=(12,8))
df.boxplot()
plt.xticks(rotation=90)
plt.title("Boxplot Before Outlier Handling")
plt.show()

# Outlier Treatment (Capping + IQR)

# Capping Insulin
Q1 = df['Insulin'].quantile(0.25)
Q3 = df['Insulin'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['Insulin'] = df['Insulin'].clip(lower=lower, upper=upper)


# Removing Outliers Using IQR (Column-wise)
cols = ['Pregnancies','Glucose','BloodPressure',
        'SkinThickness','BMI','DiabetesPedigreeFunction','Age']

for c in cols:

    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[c] >= lower) & (df[c] <= upper)]


print("\nDataset After Outlier Removal:")
print(df.info())


# Boxplot (After Outlier Handling)
plt.figure(figsize=(12,8))
df.boxplot()
plt.xticks(rotation=90)
plt.title("Boxplot After Outlier Handling")
plt.show()

# Train-Test Split

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building

knn = KNeighborsClassifier(n_neighbors=5)
naive = GaussianNB()
svm_model = SVC()
dectree = DecisionTreeClassifier(random_state=42)
ranforest = RandomForestClassifier(random_state=42)

models = [knn, naive, svm_model, dectree, ranforest]

# Training & Evaluation

for model in models:

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    print("MODEL:", model.__class__.__name__)
    print("Accuracy:", acc)

    # Classification Report
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(model.__class__.__name__)
    plt.show()