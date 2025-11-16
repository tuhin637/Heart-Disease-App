import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import pickle

print("Step 1: Loading dataset...")
df = pd.read_csv('heart_2020_cleaned.csv')
print(f"Dataset loaded! Shape: {df.shape}")

print("\nStep 2: Outlier handling...")
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

numerical_cols = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]
for col in numerical_cols:
    cap_outliers(df, col)
print("Outliers handled!")

print("\nStep 3: Label Encoding...")
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print(f"Encoded {len(label_encoders)} columns")

print("\nStep 4: Preparing features...")
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Save feature names
feature_names = X.columns.tolist()
print(f"Feature names: {feature_names}")

print("\nStep 5: Handling missing values...")
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

print("\nStep 6: Applying SMOTE...")
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"After SMOTE - X shape: {X_resampled.shape}, y shape: {y_resampled.shape}")

print("\nStep 7: Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

print("\nStep 8: Training Random Forest Model...")
best_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)
best_rf.fit(X_train, y_train)
print("Model trained!")

# Test accuracy
from sklearn.metrics import accuracy_score
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nStep 9: Saving model and encoders...")
# Save model
joblib.dump(best_rf, 'best_rf_model.pkl')
print("✓ Model saved as 'best_rf_model.pkl'")

# Save label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
print("✓ Label encoders saved as 'label_encoders.pkl'")

# Save imputer
joblib.dump(imputer, 'imputer.pkl')
print("✓ Imputer saved as 'imputer.pkl'")

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✓ Feature names saved as 'feature_names.pkl'")

print("\n" + "="*50)
print("ALL FILES SAVED SUCCESSFULLY!")
print("="*50)
print("\nFiles created:")
print("  1. best_rf_model.pkl")
print("  2. label_encoders.pkl")
print("  3. imputer.pkl")
print("  4. feature_names.pkl")
print("\nYou can now run: streamlit run app.py")