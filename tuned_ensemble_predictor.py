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

def clean_and_engineer_features(df_in):
    df = df_in.copy()
    
    # --- 1. Handle Missing Values ---
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # --- 2. Feature Engineering ---
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['CabinInitial'] = df['Cabin'].str[0].fillna('U')
    df['TicketPrefix'] = df['Ticket'].apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) > 0 else 'no_prefix')
    df['TicketPrefix'] = df['TicketPrefix'].apply(lambda x: x if x in ['no_prefix', 'a5', 'pc', 'ston/o2'] else 'other')

    # --- 3. Create Dummy Variables ---
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    categorical_cols = ['Embarked', 'Pclass', 'Title', 'CabinInitial', 'TicketPrefix']
    df_dummies = pd.get_dummies(df[categorical_cols], drop_first=True, dtype=int)
    df = pd.concat([df, df_dummies], axis=1)
    df.drop(columns=categorical_cols, inplace=True)

    # --- 4. Interaction Features ---
    df['Age*Class'] = df['Age'] * df_in['Pclass']
    df['Fare*Class'] = df['Fare'] * df_in['Pclass']

    # --- 5. Final Data Preparation ---
    df.drop(columns=['Ticket', 'Cabin', 'Name', 'PassengerId'], inplace=True, errors='ignore')
    
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'Age*Class', 'Fare*Class']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

train_data_featured = clean_and_engineer_features(train_df)
test_data_featured = clean_and_engineer_features(test_df)

X = train_data_featured.drop(columns=['Survived'])
y = train_data_featured['Survived']
X_test = test_data_featured

X_aligned, X_test_aligned = X.align(X_test, join='inner', axis=1)

# --- Hyperparameter Tuning ---
kfold = StratifiedKFold(n_splits=5)

# KNN
knn_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
knn_gs = GridSearchCV(KNeighborsClassifier(), knn_params, cv=kfold, scoring='accuracy')
knn_gs.fit(X_aligned, y)
knn_best = knn_gs.best_estimator_
print(f"Best KNN params: {knn_gs.best_params_}")

# RandomForest
rf_params = {'n_estimators': [100, 200], 'max_depth': [4, 6, 8], 'min_samples_leaf': [1, 2, 3]}
rf_gs = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=kfold, scoring='accuracy')
rf_gs.fit(X_aligned, y)
rf_best = rf_gs.best_estimator_
print(f"Best RandomForest params: {rf_gs.best_params_}")

# XGBoost
xgb_params = {'learning_rate': [0.05, 0.1], 'n_estimators': [100, 200], 'max_depth': [3, 4, 5], 'subsample': [0.8, 0.9]}
xgb_gs = GridSearchCV(XGBClassifier(random_state=42), xgb_params, cv=kfold, scoring='accuracy')
xgb_gs.fit(X_aligned, y)
xgb_best = xgb_gs.best_estimator_
print(f"Best XGBoost params: {xgb_gs.best_params_}")

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
submission.to_csv('submission_ensemble_tuned.csv', index=False)

print("\nSubmission file 'submission_ensemble_tuned.csv' created successfully!")
