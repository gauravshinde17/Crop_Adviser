import pandas as pd  
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "C:\Crop-Advisor\data\Crop_recommendation.csv"  
df = pd.read_csv(file_path)

# Display basic info
print("Dataset Loaded Successfully!")
print(df.head())  # Print first 5 rows
print(df.info())  # Check data types and missing values

# Handling missing values
print("\nChecking for missing values...")
print(df.isnull().sum())  # Display count of missing values per column

# Option 1: Fill missing values with the mean (for numerical columns)
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)


# Option 2: Drop rows with missing values (if necessary)
# df.dropna(inplace=True)

print("\nMissing values handled successfully!")
print(df.isnull().sum())  # Verify no missing values

# Feature Scaling (Normalization)
feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
scaler = MinMaxScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

print("\nFeature scaling completed successfully!")
print(df.head())  # Check scaled values

# Splitting Data for Training & Testing
X = df[feature_columns]
y = df['label']  # Assuming 'label' is the target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData splitting completed successfully!")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")
