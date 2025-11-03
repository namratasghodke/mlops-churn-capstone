import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# --- Load data ---
data_path = '../../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(data_path)

# --- Drop customerID ---
df.drop('customerID', axis=1, inplace=True)

# --- Clean and convert TotalCharges ---
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# --- Encode categorical columns ---
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- Split data ---
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# --- Scale numeric features ---
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# --- Save processed data ---
os.makedirs('../../data/processed', exist_ok=True)
X_train_scaled.to_csv('../../data/processed/X_train.csv', index=False)
X_val_scaled.to_csv('../../data/processed/X_val.csv', index=False)
X_test_scaled.to_csv('../../data/processed/X_test.csv', index=False)
y_train.to_csv('../../data/processed/y_train.csv', index=False)
y_val.to_csv('../../data/processed/y_val.csv', index=False)
y_test.to_csv('../../data/processed/y_test.csv', index=False)

print("✅ Data preprocessing completed successfully.")
