# Import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
train_df = pd.read_csv("/kaggle/input/titanic-survival-prediction/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic-survival-prediction/test.csv")
gender_submission_df = pd.read_csv("/kaggle/input/titanic-survival-prediction/gender_submission.csv")

# Feature Engineering - Title extraction
for df in [train_df, test_df]:
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\\.', expand=False)
    df['Title'] = df['Title'].replace(['Mme','Lady', 'Countess', 'Dona'], 'Mrs')
    df['Title'] = df['Title'].replace(['Capt','Master', 'Col', 'Don', 'Dr', 'Major', 
                                       'Rev', 'Sir', 'Jonkheer'], 'Mr')
    df['Title'] = df['Title'].replace(['Ms','Mlle'], 'Miss')

# Handle missing values
train_df['Age'] = train_df.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
test_df['Age'] = test_df.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Drop unnecessary columns
for df in [train_df, test_df]:
    for col in ['Embarked', 'Cabin']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

# Label encode categorical columns
for df in [train_df, test_df]:
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Title'] = LabelEncoder().fit_transform(df['Title'])

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title']
X = train_df[features]
y = train_df['Survived']
X_test = test_df[features]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validation
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}")

# Predict test set for submission
y_test_pred = model.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": y_test_pred
})
# Optional: Save submission
# submission.to_csv("submission.csv", index=False)

