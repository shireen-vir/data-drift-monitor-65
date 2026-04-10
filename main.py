import pandas as pd
import numpy as np

"""
data-drift-monitor-65: A data science tool for monitoring data drift.
"""

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        return None

def calculate_statistics(data):
    try:
        stats = data.describe()
        return stats
    except Exception as e:
        print(f"Failed to calculate statistics: {str(e)}")
        return None

def monitor_data_drift(data, threshold):
    try:
        mean = np.mean(data)
        std_dev = np.std(data)
        lower_bound = mean - threshold * std_dev
        upper_bound = mean + threshold * std_dev
        return lower_bound, upper_bound
    except Exception as e:
        print(f"Failed to monitor data drift: {str(e)}")
        return None

def main():
    file_path = "data.csv"
    threshold = 2
    data = load_data(file_path)
    if data is not None:
        stats = calculate_statistics(data)
        if stats is not None:
            print("Data Statistics:")
            print(stats)
            lower_bound, upper_bound = monitor_data_drift(data.iloc[:, 0], threshold)
            if lower_bound is not None:
                print(f"Data Drift Bounds: ({lower_bound}, {upper_bound})")

if __name__ == "__main__":
    main()