import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# Load data
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
test_passenger_ids = test_df['PassengerId']

# Combine for feature engineering
combined_df = pd.concat([train_df.drop('Survived', axis=1), test_df], ignore_index=True)

# --- Advanced Feature Engineering ---

# 1. Handle Missing Values
combined_df['Age'].fillna(combined_df['Age'].median(), inplace=True)
combined_df['Fare'].fillna(combined_df['Fare'].median(), inplace=True)
combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0], inplace=True)

# 2. Family Size and IsAlone
combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1
combined_df['IsAlone'] = (combined_df['FamilySize'] == 1).astype(int)

# 3. Fare per Person
combined_df['FarePerPerson'] = combined_df['Fare'] / combined_df['FamilySize']

# 4. Age Categories
combined_df['AgeCategory'] = pd.cut(combined_df['Age'], bins=[0, 12, 20, 40, 60, 80], labels=['Child', 'Teenager', 'Adult', 'MiddleAged', 'Senior'])

# 5. Title from Name
combined_df['Title'] = combined_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
combined_df['Title'] = combined_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined_df['Title'] = combined_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
combined_df['Title'] = combined_df['Title'].replace('Mme', 'Mrs')

# 6. Ticket Group Features
ticket_group_size = combined_df.groupby('Ticket')['Ticket'].transform('count')
combined_df['TicketGroupSize'] = ticket_group_size

# --- Separate train and test sets ---
train_data_featured = combined_df.iloc[:len(train_df)]
test_data_featured = combined_df.iloc[len(train_df):]

# Add Survived column back to train_data_featured
train_data_featured = train_data_featured.assign(Survived=train_df['Survived'])

# Group Survival Rate (calculated only on training data to prevent leakage)
ticket_survival_rate = train_data_featured.groupby('Ticket')['Survived'].mean()
train_data_featured['GroupSurvivalRate'] = train_data_featured['Ticket'].map(ticket_survival_rate)
test_data_featured['GroupSurvivalRate'] = test_data_featured['Ticket'].map(ticket_survival_rate)

# Handle cases where a ticket in test set was not in train set
train_data_featured['GroupSurvivalRate'].fillna(0.5, inplace=True)
test_data_featured['GroupSurvivalRate'].fillna(0.5, inplace=True)


# --- Final Data Preparation ---
def finalize_data(df):
    df_final = df.copy()
    
    # Convert Sex to numeric
    df_final['Sex'] = df_final['Sex'].map({'male': 0, 'female': 1})
    
    # Create dummy variables
    categorical_cols = ['Embarked', 'Pclass', 'Title', 'AgeCategory']
    df_dummies = pd.get_dummies(df_final[categorical_cols], drop_first=True, dtype=int)
    df_final = pd.concat([df_final, df_dummies], axis=1)
    df_final.drop(columns=categorical_cols, inplace=True)
    
    # Drop unnecessary columns
    df_final.drop(columns=['Ticket', 'Cabin', 'Name', 'PassengerId'], inplace=True, errors='ignore')
    
    # Scaling numerical features
    numerical_cols = [col for col in df_final.columns if df_final[col].dtype != 'object' and col not in ['Survived']]
    scaler = StandardScaler()
    # Use try-except to handle cases where a column might not exist in the test set after one-hot encoding
    try:
        df_final[numerical_cols] = scaler.fit_transform(df_final[numerical_cols])
    except KeyError as e:
        missing_col = str(e).split("'")[1]
        df_final[missing_col] = 0 # Add missing column and fill with 0
        df_final[numerical_cols] = scaler.fit_transform(df_final[numerical_cols])

    return df_final

train_final = finalize_data(train_data_featured)
test_final = finalize_data(test_data_featured)


# Align columns
X = train_final.drop(columns=['Survived'])
y = train_final['Survived']
X_test = test_final

X_aligned, X_test_aligned = X.align(X_test, join='inner', axis=1)

# --- Hyperparameter Tuning (using previously found best params) ---
knn_best = KNeighborsClassifier(n_neighbors=5, weights='uniform')
rf_best = RandomForestClassifier(max_depth=8, min_samples_leaf=1, n_estimators=100, random_state=42)
xgb_best = XGBClassifier(learning_rate=0.05, max_depth=5, n_estimators=200, subsample=0.9, random_state=42)

# --- Tuned Ensemble Model ---
voting_clf_tuned = VotingClassifier(
    estimators=[('knn', knn_best), ('rf', rf_best), ('xgb', xgb_best)],
    voting='hard'
)
voting_clf_tuned.fit(X_aligned, y)

# --- Prediction ---
predictions = voting_clf_tuned.predict(X_test_aligned)

# --- Create Submission File ---
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': predictions
})
submission.to_csv('submission_advanced_features.csv', index=False)

print("Submission file 'submission_advanced_features.csv' created successfully!")
