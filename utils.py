import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    """
    Data Drift Monitor Tool.

    This tool is designed to monitor and detect data drift in datasets.
    It calculates various statistics and metrics to determine if there 
    is a significant difference between the training and testing datasets.
    """
    # Load the dataset
    df = pd.read_csv('data.csv')

    # Split the dataset into training and testing sets
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_train)

    # Print the results
    print(f"Mean Squared Error: {mse}")

    # Calculate the difference in means between the training and testing sets
    mean_diff = np.mean(X_train) - np.mean(X_test)

    # Print the results
    print(f"Mean Difference: {mean_diff}")

if __name__ == "__main__":
    main()