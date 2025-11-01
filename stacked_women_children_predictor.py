import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- Level 0 Model: Women and Children Classifier ---
class WomenAndChildrenClassifier(BaseEstimator, ClassifierMixin):
    """Predicts survival based on the 'women and children first' rule."""
    def __init__(self, child_age_threshold=12):
        self.child_age_threshold = child_age_threshold

    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        # Predict 1 (survived) if Sex is 1 (female) OR Age <= threshold
        predictions = ((X['Sex'] == 1) | (X['Age'] <= self.child_age_threshold)).astype(int)
        return predictions.values

# --- Data Loading and Feature Engineering ---
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
test_passenger_ids = test_df['PassengerId']

def clean_and_engineer_features(df_in):
    df = df_in.copy()
    
    # Basic cleaning
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Convert Sex to numeric first
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Dummy Variables
    categorical_cols = ['Embarked', 'Pclass', 'Title']
    df_dummies = pd.get_dummies(df[categorical_cols], drop_first=True, dtype=int)
    df = pd.concat([df, df_dummies], axis=1)
    df.drop(columns=categorical_cols, inplace=True)

    # Final Data Preparation
    df.drop(columns=['Ticket', 'Cabin', 'Name', 'PassengerId'], inplace=True, errors='ignore')
    
    return df

# Process features for both datasets
train_featured = clean_and_engineer_features(train_df)
test_featured = clean_and_engineer_features(test_df)

# --- Stacking: Create Level 1 Feature ---
# Instantiate and "train" the women and children model
wc_model = WomenAndChildrenClassifier(child_age_threshold=12)
wc_model.fit(train_featured)

# Get predictions from the base model
train_wc_preds = wc_model.predict(train_featured)
test_wc_preds = wc_model.predict(test_featured)

# Add the predictions as a new feature
train_featured['WomenAndChildrenPrediction'] = train_wc_preds
test_featured['WomenAndChildrenPrediction'] = test_wc_preds

# --- Scaling (after creating all features) ---
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
scaler = StandardScaler()
train_featured[numerical_cols] = scaler.fit_transform(train_featured[numerical_cols])
test_featured[numerical_cols] = scaler.transform(test_featured[numerical_cols])


# --- Level 1 Model: The Ensemble ---
# Align columns after adding the new feature
X = train_featured
y = train_df['Survived']
X_test = test_featured

X_aligned, X_test_aligned = X.align(X_test, join='inner', axis=1)

# --- Model Validation ---
X_train, X_val, y_train, y_val = train_test_split(X_aligned, y, test_size=0.2, random_state=42, stratify=y)

# Define the untuned classifiers
knn = KNeighborsClassifier(n_neighbors=6)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(random_state=42, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.9)

voting_clf = VotingClassifier(
    estimators=[('knn', knn), ('rf', rf), ('xgb', xgb)],
    voting='hard'
)

# Train and evaluate on the validation set
voting_clf.fit(X_train, y_train)
y_pred_val = voting_clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy of the Tuned Ensemble: {accuracy:.4f}")


# --- Final Training and Prediction ---
# Retrain on the full dataset
voting_clf.fit(X_aligned, y)
final_predictions = voting_clf.predict(X_test_aligned)

# --- Create Submission File ---
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': final_predictions
})
submission.to_csv('submission_stacked_wc.csv', index=False)

print("Submission file 'submission_stacked_wc.csv' created successfully!")
