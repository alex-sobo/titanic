
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the training data
try:
    df = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    print("Error: 'input/train.csv' not found. Make sure the file is in the 'input' directory.")
    exit()

# --- 1. Handle Missing Values ---

# Fill missing 'Age' values with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the most frequent port
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Fill missing 'Fare' values with the median fare
df['Fare'].fillna(df['Fare'].median(), inplace=True)


# --- 2. Create Dummy Variables for Categorical Features ---

# Convert 'Sex' to a numerical format (0 for male, 1 for female)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked' and 'Pclass'
df = pd.get_dummies(df, columns=['Embarked', 'Pclass'], drop_first=True)


# --- 3. Normalize Numerical Columns ---

# Select numerical columns for scaling
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the data
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# --- 4. Final Data Preparation ---

# Drop columns that are not needed for the model
# We keep 'Name' as requested, and 'PassengerId' for identification
df_cleaned = df.drop(columns=['Ticket', 'Cabin'])

# Save the cleaned data to a new CSV file
output_path = './input/train_cleaned.csv'
df_cleaned.to_csv(output_path, index=False)

print(f"Data cleaning complete. Cleaned data saved to '{output_path}'")
print("\nFirst 5 rows of the cleaned data:")
print(df_cleaned.head())
