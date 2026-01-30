# 1. Importing Libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Loading Dataset
df = pd.read_csv("stroke.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())
print(df.columns)

df.drop(columns=["id"], inplace=True)

df["bmi"].fillna(df["bmi"].mean(), inplace=True)

# Stroke Distribution
plt.figure()
df["stroke"].value_counts().plot(kind="bar")
plt.title("Stroke vs Non-Stroke Distribution")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Age vs Stroke
plt.figure()
df.boxplot(column="age", by="stroke")
plt.title("Age Distribution by Stroke")
plt.suptitle("")
plt.xlabel("Stroke")
plt.ylabel("Age")
plt.show()

# BMI Distribution by Stroke
plt.figure()

df[df["stroke"] == 0]["bmi"].plot(kind="hist", alpha=0.6, bins=30, label="No Stroke")
df[df["stroke"] == 1]["bmi"].plot(kind="hist", alpha=0.6, bins=30, label="Stroke")

plt.title("BMI Distribution by Stroke")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Correlation Heatmap
corr = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()

# Gender vs Stroke
gender_stroke = pd.crosstab(df["gender"], df["stroke"])

gender_stroke.plot(kind="bar")
plt.title("Gender vs Stroke")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(["No Stroke", "Stroke"])
plt.show()

# Smoking Status vs Stroke
smoke_stroke = pd.crosstab(df["smoking_status"], df["stroke"])

smoke_stroke.plot(kind="bar")
plt.title("Smoking Status vs Stroke")
plt.xlabel("Smoking Status")
plt.ylabel("Count")
plt.legend(["No Stroke", "Stroke"])
plt.show()

# 3. Encoding Categorical Features
categorical_cols = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 4. Feature & Target Split
X = df.drop(columns=["stroke"])
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 5. Models (with Pipelines)
models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),
    
    "Naive Bayes": GaussianNB(),
    
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ]),
    
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# =========================
# 6. Training & Evaluation
# =========================
accuracies={}
for name, model in models.items():
    print(f"\nModel: {name}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    accuracies[name] = acc

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues")
    plt.title(name)
    plt.show()
    print("-" * 60)

# Model Comparison
plt.figure()
plt.bar(accuracies.keys(), accuracies.values())

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.ylim(0, 1)
plt.show()



