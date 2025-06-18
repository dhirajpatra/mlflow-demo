# preprocess.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # For handling missing values

def preprocess_data():
    """
    Loads the Iris dataset, performs robust preprocessing and checks,
    and splits the data into training and testing sets.

    Returns:
        tuple: X_train, X_test, y_train, y_test
               (Pandas DataFrames/Series)
    """
    print("Starting data preprocessing...")

    # Load the dataset
    try:
        iris = load_iris(as_frame=True)
        df = iris.frame
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None, None

    # --- Initial Data Inspection ---
    print("\n--- Initial Data Inspection ---")
    print(f"Original DataFrame shape: {df.shape}")
    print("\nFirst 5 rows of the DataFrame:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Separate features (X) and target (y)
    if "target" not in df.columns:
        print("Error: 'target' column not found in the DataFrame.")
        return None, None, None, None
    X = df.drop("target", axis=1)
    y = df["target"]
    print(f"\nFeatures (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")

    # --- Data Quality Checks and Handling ---

    # 1. Check for Duplicate Rows
    print("\n--- Data Quality Checks ---")
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        print(f"Warning: Found {num_duplicates} duplicate rows. Removing them.")
        df.drop_duplicates(inplace=True)
        # Re-separate X and y after dropping duplicates if necessary,
        # though for Iris, it's unlikely to change the target.
        X = df.drop("target", axis=1)
        y = df["target"]
        print(f"DataFrame shape after removing duplicates: {df.shape}")
    else:
        print("No duplicate rows found.")

    # 2. Check for Missing Values
    print("\nChecking for missing values...")
    missing_values_summary = X.isnull().sum()
    if missing_values_summary.sum() > 0:
        print("Missing values found per column:")
        print(missing_values_summary[missing_values_summary > 0])
        print("Imputing missing values using the mean strategy.")
        # For simplicity, using SimpleImputer. For real data, might need more complex strategies.
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        print("Missing values imputation complete.")
    else:
        print("No missing values found.")

    # 3. Check for Outliers (Simple IQR-based check for numerical columns)
    print("\nPerforming a simple outlier check (IQR method)...")
    numerical_cols = X.select_dtypes(include=np.number).columns
    outlier_found = False
    for col in numerical_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)]
        if not outliers.empty:
            print(f"  Column '{col}': Found {len(outliers)} potential outliers.")
            outlier_found = True
    if not outlier_found:
        print("No significant outliers detected by simple IQR method.")
    # Note: Outlier handling (removal/transformation) depends on the context and domain.
    # For now, just detecting.

    # 4. Check Data Types (Ensure numerical columns are indeed numerical)
    print("\nChecking data types...")
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"Warning: Column '{col}' is not numeric. Consider converting or encoding.")
    print("Data type check complete.")

    # --- Feature Scaling (Standardization) ---
    print("\n--- Feature Scaling (Standardization) ---")
    print("Applying StandardScaler to numerical features.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    print("Features scaled successfully.")
    print("\nFirst 5 rows of scaled features:")
    print(X.head())

    # --- Data Splitting ---
    print("\n--- Data Splitting ---")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Data split into training and testing sets with a 80/20 ratio.")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print("\nDistribution of target classes in training set:")
        print(y_train.value_counts(normalize=True))
        print("\nDistribution of target classes in test set:")
        print(y_test.value_counts(normalize=True))
    except Exception as e:
        print(f"Error during data splitting: {e}")
        return None, None, None, None

    print("\nData preprocessing complete!")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()

    if X_train is not None:
        print("\nExample: Displaying a portion of X_train after preprocessing:")
        print(X_train.head())