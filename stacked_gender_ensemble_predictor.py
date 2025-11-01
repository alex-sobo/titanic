import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# --- Level 0 Model: Gender-Based Classifier ---
class GenderBasedClassifier(BaseEstimator, ClassifierMixin):
    """A simple classifier that predicts survival based on gender."""
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        # Predict 1 (survived) if Sex is 1 (female), otherwise 0
        return X['Sex'].apply(lambda x: 1 if x == 1 else 0).values

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
    
    # Convert Sex to numeric first, as it's needed for the GenderBasedClassifier
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['CabinInitial'] = df['Cabin'].str[0].fillna('U')
    df['TicketPrefix'] = df['Ticket'].apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) > 0 else 'no_prefix')
    df['TicketPrefix'] = df['TicketPrefix'].apply(lambda x: x if x in ['no_prefix', 'a5', 'pc', 'ston/o2'] else 'other')
    
    # Dummy Variables
    categorical_cols = ['Embarked', 'Pclass', 'Title', 'CabinInitial', 'TicketPrefix']
    df_dummies = pd.get_dummies(df[categorical_cols], drop_first=True, dtype=int)
    df = pd.concat([df, df_dummies], axis=1)
    df.drop(columns=categorical_cols, inplace=True)

    # Interaction Features
    df['Age*Class'] = df['Age'] * df_in['Pclass']
    df['Fare*Class'] = df['Fare'] * df_in['Pclass']

    # Final Data Preparation
    df.drop(columns=['Ticket', 'Cabin', 'Name', 'PassengerId'], inplace=True, errors='ignore')
    
    # Scaling
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'Age*Class', 'Fare*Class']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

# Process features for both datasets
train_featured = clean_and_engineer_features(train_df)
test_featured = clean_and_engineer_features(test_df)

# --- Stacking: Create Level 1 Feature ---
# Instantiate and "train" the gender-based model
gender_model = GenderBasedClassifier()
gender_model.fit(train_featured)

# Get predictions from the gender model
train_gender_preds = gender_model.predict(train_featured)
test_gender_preds = gender_model.predict(test_featured)

# Add the predictions as a new feature
train_featured['GenderModelPrediction'] = train_gender_preds
test_featured['GenderModelPrediction'] = test_gender_preds

# --- Level 1 Model: The Ensemble ---
# Align columns after adding the new feature
X = train_featured
y = train_df['Survived'] # Use original Survived column
X_test = test_featured

X_aligned, X_test_aligned = X.align(X_test, join='inner', axis=1)

# Define the untuned classifiers (your best-performing setup)
knn = KNeighborsClassifier(n_neighbors=6)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(random_state=42, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.9)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('knn', knn), ('rf', rf), ('xgb', xgb)],
    voting='hard'
)

# Train the main ensemble on the augmented data
voting_clf.fit(X_aligned, y)

# --- Final Prediction ---
final_predictions = voting_clf.predict(X_test_aligned)

# --- Create Submission File ---
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': final_predictions
})
submission.to_csv('submission_stacked_gender.csv', index=False)

print("Submission file 'submission_stacked_gender.csv' created successfully!")
