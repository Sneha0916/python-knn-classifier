# python-knn-classifier
This Python program uses the K-Nearest Neighbors (KNN) algorithm to predict whether a person has diabetes based on medical attributes from a dataset. It includes data loading, preprocessing (feature scaling), model training, evaluation, and user input prediction.
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv(r"C:\Users\sneha\Downloads\diabetes.csv")

# Separate features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



# Feature Scaling (Min-Max Scaling)
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)  # Fit and transform training data
x_test_scaled = scaler.transform(x_test)        # Transform test data


# Initialize and fit the KNN classifier
knn = KNeighborsClassifier(n_neighbors=9, metric="euclidean")
knn.fit(x_train_scaled, y_train)

# Make predictions on the test set
y_test_pred = knn.predict(x_test_scaled)

# Model Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))


# Dynamically enter feature values for prediction
print("\nPlease enter values for the features to predict diabetes:")
user_input = []
for feature in X.columns:  # Ensure we use the column names from X (features) 
    value = float(input(f"{feature}: "))
    user_input.append(value)



# Create a DataFrame from user input for prediction and make prediction
user_data = pd.DataFrame([user_input], columns=X.columns)  # Ensure feature names are the same
user_data_scaled = scaler.transform(user_data)  # Scale the user input


# Make the prediction
user_prediction = knn.predict(user_data_scaled)

print("\nPrediction for the entered feature values:")
print(user_prediction)
