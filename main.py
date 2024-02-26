import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.utils import parallel_backend


# Function to create a menu for the program
def menu():
    print("\nMachine Learning Project Menu:")
    print("1. Explore Dataset")
    print("2. Evaluate Nearest Neighbor Baseline (k=1 and k=3)")
    print("3. Evaluate Logistic Regression with SGD and Validation")
    print("4. Evaluate SVM with Validation")
    print("5. Exit\n")

# Function to replace missing values based on column type

def replace_missing_values(df):
    # Print the number of missing values before replacement
    print("Number of missing values before replacement:")
    print(df.isnull().sum())

    for column in df.columns:
        if df[column].dtype == 'object':
            # Replace missing values in categorical columns with mode
            mode_value = df[column].mode()[0]
            df.loc[df[column].isnull(), column] = mode_value
        else:
            # Replace missing values in numeric columns with mean
            mean_value = df[column].mean()

            # Explicitly cast to compatible dtype
            if df[column].dtype == 'int64':
                mean_value = int(mean_value)

            df.loc[df[column].isnull(), column] = mean_value

    # Print the number of missing values after replacement
    print("\nNumber of missing values after replacement:")
    print(df.isnull().sum())

    return df  # Return the modified DataFrame
def perform_eda(df,subset_size=100):
    # Print the shape and head of the DataFrame to check its contents
    print("DataFrame Shape:", df.shape)
    print("DataFrame Head:")
    print(df.head())

    # Descriptive statistics for each column
    for column in df.columns:
        print(f"\n{column.capitalize()}:")
        print(f"   - Count: {df[column].count()}")
        print(f"   - Unique values: {df[column].nunique()}")
        print(f"   - Top value: {df[column].mode().iloc[0]}")
        print(f"   - Frequency of top value: {df[column].value_counts().iloc[0]}")

        if df[column].dtype in ['int64', 'float64']:
            print(f"   - Mean: {df[column].mean()}")
            print(f"   - Standard deviation: {df[column].std()}")
            print(f"   - Minimum: {df[column].min()}")
            print(f"   - 25th percentile (Q1): {df[column].quantile(0.25)}")
            print(f"   - Median (50th percentile or Q2): {df[column].median()}")
            print(f"   - 75th percentile (Q3): {df[column].quantile(0.75)}")
            print(f"   - Maximum: {df[column].max()}")
        else:
            print("   - No statistics for non-numeric columns")

    # Visualizations

    # Heatmap for Numeric Data
    # Heatmap for Numeric Data Correlation
    numeric_data = df.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Data')
    plt.show()


    plt.figure(figsize=(8, 5))
    sns.countplot(x='income', data=df)
    plt.title('Distribution of Income')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(x='age', data=df, bins=20, kde=True)
    plt.title('Age Distribution')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='workclass', data=df, hue='income')
    plt.title('Workclass Distribution by Income')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='native-country', y='count', data=df.groupby('native-country').size().reset_index(name='count'))
    plt.title('Native Country Distribution')
    plt.xticks(rotation=45)
    plt.show()


    plt.figure(figsize=(12, 6))
    sns.countplot(x='sex', data=df, hue='income')
    plt.title('Gender Distribution by Income')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='income', y='hours-per-week', data=df)
    plt.title('Hours Per Week Distribution by Income')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='education', data=df, hue='income')
    plt.title('Education Level Distribution by Income')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(14, 8))
    sns.countplot(x='occupation', data=df, hue='income')
    plt.title('Occupation Distribution by Income')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='relationship', data=df, hue='income')
    plt.title('Relationship Status Distribution by Income')
    plt.xticks(rotation=45)
    plt.show()

# Function to convert categorical columns to numeric using one-hot encoding
def convert_categorical_to_numeric(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=True)
    return df_encoded

# Function to create 'race' column
def create_race_column(df):
    df['race'] = df[['race_ White', 'race_ Black']].idxmax(axis=1).apply(lambda x: x.split('_')[-1])
    return df

# Function to split the dataset into training and testing sets

def split_data(df):
    X = df.drop('income_ >50K', axis=1)
    y = df['income_ >50K']

    # Load the test data from 'adult.test'
    test_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'income']

    # Replace "?" with NaN while reading the test data
    df_test = pd.read_csv('adult.test', names=test_columns, na_values=' ?', skiprows=1)

    # Replace missing values in the test set
    df_test = replace_missing_values(df_test)

    # Convert categorical columns to numeric using one-hot encoding for the test set
    df_test_encoded = convert_categorical_to_numeric(df_test)

    # Make sure the test set has the same columns as the training set
    df_test_encoded = df_test_encoded.reindex(columns=X.columns, fill_value=0)

    # Ensure 'income_ >50K' column is present in the test set
    if 'income_ >50K' not in df_test_encoded.columns:
        df_test_encoded['income_ >50K'] = 0  # Adding a placeholder column if missing

    X_train, y_train = X, y
    X_test, y_test = df_test_encoded.drop('income_ >50K', axis=1), df_test_encoded['income_ >50K']

    return X_train, X_test, y_train, y_test

def evaluate_baseline(X_train, X_test, y_train, y_test, k, subset_size=100):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Randomly select a subset of the testing data
    subset_indices = np.random.choice(len(X_test), subset_size, replace=False)
    X_test_subset = X_test.iloc[subset_indices, :]

    # Get distances of k-nearest neighbors for the subset
    distances, neighbors = knn.kneighbors(X_test_subset)

    # Create a scatter plot to visualize distances
    plt.figure(figsize=(12, 6))
    for i in range(len(X_test_subset)):
        plt.scatter([i] * k, distances[i], c='blue', marker='o', cmap='viridis')

    plt.title(f'Distances for KNN (k={k}) - Subset of {subset_size} Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

    # Predictions and evaluation on the entire testing set
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)

    print(f"Performance Evaluation (k={k}):")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

def evaluate_randomForest(X_train, X_test, y_train, y_test):
    # Random Forest
    print("\nEvaluating Random Forest:")
    rf_params = {'n_estimators': [50, 100, 150, 200]}
    rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=3)
    rf_grid.fit(X_train, y_train)

    rf_pred = rf_grid.best_estimator_.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred, zero_division=1)

    print(f"Random Forest - Best Hyperparameters: {rf_grid.best_params_}")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print("Classification Report:")
    print(rf_report)


def evaluate_sgd(X_train, X_test, y_train, y_test):
    print("\nEvaluating Stochastic Gradient Descent (SGD):")

    # Update 'loss' parameter to 'log_loss'
    sgd = SGDClassifier(loss='log', alpha=0.0001, max_iter=1000, random_state=42)
    sgd.fit(X_train, y_train)

    sgd_pred = sgd.predict(X_test)
    sgd_accuracy = accuracy_score(y_test, sgd_pred)
    sgd_report = classification_report(y_test, sgd_pred, zero_division=1)

    print(f"Accuracy: {sgd_accuracy:.4f}")
    print("Classification Report:")
    print(sgd_report)


def evaluate_logistic_regression(X_train, X_test, y_train, y_test):
    print("\nEvaluating Logistic Regression:")

    # Create and fit the logistic regression model
    logistic_regression = LogisticRegression(random_state=42)
    logistic_regression.fit(X_train, y_train)

    # Predictions and evaluation
    logistic_regression_pred = logistic_regression.predict(X_test)
    logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_pred)
    logistic_regression_report = classification_report(y_test, logistic_regression_pred, zero_division=1)

    print(f"Accuracy: {logistic_regression_accuracy:.4f}")
    print("Classification Report:")
    print(logistic_regression_report)


#correct for log loss
def evaluate_logistic_regression_sgd(X_train, X_test, y_train, y_test):
    print("\nEvaluating Logistic Regression with SGD:")

    # Initialize SGDClassifier for logistic regression
    logistic_regression_sgd = SGDClassifier(loss='log_loss', alpha=0.0001, max_iter=10000, random_state=42)

    # Fit the logistic regression model using SGD on the training data
    logistic_regression_sgd.fit(X_train, y_train)

    # Predictions on the test data
    logistic_regression_sgd_pred = logistic_regression_sgd.predict(X_test)

    # Evaluate the performance of the logistic regression model with SGD
    logistic_regression_sgd_accuracy = accuracy_score(y_test, logistic_regression_sgd_pred)
    logistic_regression_sgd_report = classification_report(y_test, logistic_regression_sgd_pred, zero_division=1)

    # Print the results
    print(f"Accuracy: {logistic_regression_sgd_accuracy:.4f}")
    print("Classification Report:")
    print(logistic_regression_sgd_report)


def evaluate_logistic_regression_sgd_with_validation(X_train, X_test, y_train, y_test, alpha_values, validation_size=0.3):
    validation_accuracies = []

    # Split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)

    for alpha in alpha_values:
        print(f"\nEvaluating Logistic Regression with SGD (alpha={alpha}):")

        # Initialize SGDClassifier for logistic regression
        logistic_regression_sgd = SGDClassifier(loss='log_loss', alpha=alpha, max_iter=10000, random_state=42)

        # Fit the logistic regression model using SGD on the training data
        logistic_regression_sgd.fit(X_train, y_train)

        # Predictions on the validation set
        logistic_regression_sgd_val_pred = logistic_regression_sgd.predict(X_val)

        # Evaluate the performance on the validation set
        logistic_regression_sgd_val_accuracy = accuracy_score(y_val, logistic_regression_sgd_val_pred)
        validation_accuracies.append(logistic_regression_sgd_val_accuracy)

        # Print the results for the validation set
        print(f"Validation Set - Accuracy (alpha={alpha}): {logistic_regression_sgd_val_accuracy:.4f}")

    # Plot the validation accuracies for different alpha values
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, validation_accuracies, marker='o', linestyle='-')
    plt.title('Validation Accuracy for Different Alpha Values')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.xscale('log')  # Use log scale for better visibility
    plt.show()

    # Now, train the selected model with the best alpha on the entire training set
    best_alpha = alpha_values[np.argmax(validation_accuracies)]
    best_model = SGDClassifier(loss='log_loss', alpha=best_alpha, max_iter=10000, random_state=42)
    best_model.fit(X_train, y_train)

    # Predictions on the test set using the best model
    logistic_regression_sgd_test_pred = best_model.predict(X_test)

    # Evaluate the performance on the test set
    logistic_regression_sgd_test_accuracy = accuracy_score(y_test, logistic_regression_sgd_test_pred)
    logistic_regression_sgd_test_report = classification_report(y_test, logistic_regression_sgd_test_pred, zero_division=1)

    # Print the results for the test set using the best model
    print(f"\nBest Model (alpha={best_alpha}) - Test Set Accuracy: {logistic_regression_sgd_test_accuracy:.4f}")
    print("Best Model - Test Set Classification Report:")
    print(logistic_regression_sgd_test_report)





def evaluate_svm_with_validation(X_train, X_test, y_train, y_test, C_values, validation_size=0.3):
    validation_accuracies = []

    # Split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)

    for C in C_values:
        print(f"\nEvaluating SVM (C={C}):")

        # Initialize SVM classifier
        svm_classifier = SVC(C=C, kernel='rbf', gamma='scale', random_state=42)

        # Fit the SVM model on the training data
        svm_classifier.fit(X_train, y_train)

        # Predictions on the validation set
        svm_val_pred = svm_classifier.predict(X_val)

        # Evaluate the performance on the validation set
        svm_val_accuracy = accuracy_score(y_val, svm_val_pred)
        validation_accuracies.append(svm_val_accuracy)

        # Print the results for the validation set
        print(f"Validation Set - Accuracy (C={C}): {svm_val_accuracy:.4f}")

    # Plot the validation accuracies for different C values
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, validation_accuracies, marker='o', linestyle='-')
    plt.title('Validation Accuracy for Different C Values (SVM)')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.xscale('log')  # Use log scale for better visibility
    plt.show()

    # Now, train the selected model with the best C on the entire training set
    best_C = C_values[np.argmax(validation_accuracies)]
    best_svm_model = SVC(C=best_C, kernel='rbf', gamma='scale', random_state=42)
    best_svm_model.fit(X_train, y_train)

    # Predictions on the test set using the best model
    svm_test_pred = best_svm_model.predict(X_test)

    # Evaluate the performance on the test set
    svm_test_accuracy = accuracy_score(y_test, svm_test_pred)
    svm_test_report = classification_report(y_test, svm_test_pred, zero_division=1)

    # Print the results for the test set using the best model
    print(f"\nBest SVM Model (C={best_C}) - Test Set Accuracy: {svm_test_accuracy:.4f}")
    print("Best SVM Model - Test Set Classification Report:")
    print(svm_test_report)


from sklearn.linear_model import SGDClassifier

def evaluate_svm_with_validation_hinge(X_train, X_test, y_train, y_test, alpha_values, validation_size=0.3):
    validation_accuracies = []

    # Split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)

    for alpha in alpha_values:
        print(f"\nEvaluating SVM with Hinge Loss (alpha={alpha}):")

        # Initialize SGDClassifier for linear SVM (hinge loss)
        svm_classifier = SGDClassifier(loss='hinge', alpha=alpha, max_iter=10000, random_state=42)

        # Fit the SVM model on the training data
        svm_classifier.fit(X_train, y_train)

        # Predictions on the validation set
        svm_val_pred = svm_classifier.predict(X_val)

        # Evaluate the performance on the validation set
        svm_val_accuracy = accuracy_score(y_val, svm_val_pred)
        validation_accuracies.append(svm_val_accuracy)

        # Print the results for the validation set
        print(f"Validation Set - Accuracy (alpha={alpha}): {svm_val_accuracy:.4f}")

    # Plot the validation accuracies for different alpha values
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, validation_accuracies, marker='o', linestyle='-')
    plt.title('Validation Accuracy for Different Alpha Values (SVM with Hinge Loss)')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.xscale('log')  # Use log scale for better visibility
    plt.show()

    # Now, train the selected model with the best alpha on the entire training set
    best_alpha = alpha_values[np.argmax(validation_accuracies)]
    best_svm_model = SGDClassifier(loss='hinge', alpha=best_alpha, max_iter=10000, random_state=42)
    best_svm_model.fit(X_train, y_train)

    # Predictions on the test set using the best model
    svm_test_pred = best_svm_model.predict(X_test)

    # Evaluate the performance on the test set
    svm_test_accuracy = accuracy_score(y_test, svm_test_pred)
    svm_test_report = classification_report(y_test, svm_test_pred, zero_division=1)

    # Print the results for the test set using the best model
    print(f"\nBest SVM Model with Hinge Loss (alpha={best_alpha}) - Test Set Accuracy: {svm_test_accuracy:.4f}")
    print("Best SVM Model - Test Set Classification Report:")
    print(svm_test_report)

# Load the data into a DataFrame
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']

# Replace "?" with NaN while reading the data
df = pd.read_csv('adult.data', names=columns, na_values=' ?')

# Replace missing values
df = replace_missing_values(df)
# Convert categorical columns to numeric using one-hot encoding
df_encoded = convert_categorical_to_numeric(df)

# Print the column names to check if 'income' is present
print("Column Names after One-Hot Encoding:")
print(df_encoded.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(df_encoded)

warnings.simplefilter(action='ignore', category=FutureWarning)

while True:
    # Display menu
    menu()

    choice = input("Enter your choice (1-5): ")

    if choice == '1':
        print()
        perform_eda(df)
    elif choice == '2':
        # Evaluate nearest neighbor baseline model
        print("\nEvaluating Nearest Neighbor Baseline:")
        k_values = [1, 3]
        for k in k_values:
            evaluate_baseline(X_train, X_test, y_train, y_test, k)

        print()
    elif choice == '3':
        # Evaluate additional machine learning models
        #evaluate_randomForest(X_train, X_test, y_train, y_test)
        #evaluate_sgd(X_train, X_test, y_train, y_test)
        #evaluate_logistic_regression(X_train, X_test, y_train, y_test)
        #correct for log loss
        #evaluate_logistic_regression_sgd(X_train, X_test, y_train, y_test)
        print("\nEvaluating Logistic Regression with SGD and Validation:")
        alpha_values_to_try = [0.0001, 0.001, 0.01, 0.1]
        evaluate_logistic_regression_sgd_with_validation(X_train, X_test, y_train, y_test, alpha_values_to_try)

    elif choice == '4':
        print("Evaluating SVM with validation!")
        # Define a list of C values to try
        C_values_svm = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

        # Call the SVM evaluation function
        #evaluate_svm_with_validation(X_train, X_test, y_train, y_test, C_values_svm)
        evaluate_svm_with_validation_hinge(X_train, X_test, y_train, y_test, C_values_svm)
    elif choice== '5':
        print("Exiting the program. Goodbye!")
        break
    else:
        print("Invalid choice. Please enter a valid option.")
