import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib  # To save the model
# Load the dataset
file_path = 'cardio.csv'
df = pd.read_csv(file_path)

# Split the data into features and target
X = df.drop(columns=['target'])
y = df['target']
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define the Random Forest Classifier class
class RandomForestCustom:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Bootstrap sampling (sampling with replacement)
        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y.iloc[sample_indices].values  # Ensure correct indexing by using iloc

            # Train a decision tree (simplified decision tree for now)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X):
        # Get predictions from all trees and take the majority vote
        tree_preds = np.zeros((len(X), len(self.trees)))

        for i, tree in enumerate(self.trees):
            tree_preds[:, i] = tree.predict(X)

        # Majority vote
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=tree_preds)
        return majority_votes


# Cross-validation to find best hyperparameters (n_estimators and max_depth)
def cross_validate_rf(X_train, y_train, n_estimators_values, max_depth_values):
    best_n_estimators = n_estimators_values[0]
    best_max_depth = max_depth_values[0]
    best_accuracy = 0

    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            accuracies = []

            for train_idx, val_idx in kf.split(X_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[
                    val_idx]  # Use .iloc for correct indexing

                rf_model = RandomForestCustom(n_estimators=n_estimators, max_depth=max_depth)
                rf_model.fit(X_fold_train, y_fold_train)
                y_val_pred = rf_model.predict(X_fold_val)
                fold_accuracy = accuracy_score(y_fold_val, y_val_pred)
                accuracies.append(fold_accuracy)

            mean_accuracy = np.mean(accuracies)

            if mean_accuracy > best_accuracy:
                best_n_estimators = n_estimators
                best_max_depth = max_depth
                best_accuracy = mean_accuracy

    return best_n_estimators, best_max_depth, best_accuracy


# Define hyperparameter values for cross-validation
n_estimators_values = [50, 100, 200]
max_depth_values = [None, 10, 20]

# Perform cross-validation
best_n_estimators, best_max_depth, best_cv_accuracy = cross_validate_rf(X_train_scaled, y_train, n_estimators_values,
                                                                        max_depth_values)

# Train the model using the best hyperparameters
rf_model = RandomForestCustom(n_estimators=best_n_estimators, max_depth=best_max_depth)
rf_model.fit(X_train_scaled, y_train)

# Test the model
y_test_pred = rf_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print results
# print(f'Best n_estimators: {best_n_estimators}, Best max_depth: {best_max_depth}, Cross-Validation Accuracy: {best_cv_accuracy:.4f}')
print(f'Test Accuracy with best n_estimators={best_n_estimators} and max_depth={best_max_depth}: {test_accuracy:.4f}')

# Classification report
report = classification_report(y_test, y_test_pred)



joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler as well
print("Model saved to 'random_forest_model.pkl'")
