
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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
    # FamilySize and IsAlone
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Title from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Cabin initial
    df['CabinInitial'] = df['Cabin'].str[0].fillna('U')

    # Ticket prefix
    df['TicketPrefix'] = df['Ticket'].apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) > 0 else 'no_prefix')
    df['TicketPrefix'] = df['TicketPrefix'].apply(lambda x: x if x in ['no_prefix', 'a5', 'pc', 'ston/o2'] else 'other')

    # --- 3. Create Dummy Variables ---
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    categorical_cols = ['Embarked', 'Pclass', 'Title', 'CabinInitial', 'TicketPrefix']
    df_dummies = pd.get_dummies(df[categorical_cols], drop_first=True, dtype=int)
    df = pd.concat([df, df_dummies], axis=1)
    df.drop(columns=categorical_cols, inplace=True)

    # --- 4. Interaction Features ---
    df['Age*Class'] = df['Age'] * df_in['Pclass'] # Using original Pclass
    df['Fare*Class'] = df['Fare'] * df_in['Pclass']

    # --- 5. Final Data Preparation ---
    df.drop(columns=['Ticket', 'Cabin', 'Name', 'PassengerId'], inplace=True, errors='ignore')
    
    # Scaling numerical features
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'Age*Class', 'Fare*Class']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

# Clean and engineer features for train and test data
train_data_featured = clean_and_engineer_features(train_df)
test_data_featured = clean_and_engineer_features(test_df)

# Align columns
X = train_data_featured.drop(columns=['Survived'])
y = train_data_featured['Survived']
X_test = test_data_featured

X_aligned, X_test_aligned = X.align(X_test, join='inner', axis=1) # Use inner join to keep only common columns

# --- Model Training ---
# Define classifiers
knn = KNeighborsClassifier(n_neighbors=6)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(random_state=42, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.9)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('knn', knn), ('rf', rf), ('xgb', xgb)],
    voting='hard'
)

# Train the Voting Classifier
voting_clf.fit(X_aligned, y)

# --- Prediction ---
predictions = voting_clf.predict(X_test_aligned)

# --- Create Submission File ---
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': predictions
})
submission.to_csv('submission_ensemble.csv', index=False)

print("Submission file 'submission_ensemble.csv' created successfully!")
