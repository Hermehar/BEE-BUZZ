import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
!pip uninstall scikit-learn tpot
!pip install scikit-learn==1.2.2 tpot
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
!pip install tpot
from tpot import TPOTClassifier
from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection

#EDA before Pre-processing
df = pd.read_csv('/content/drive/MyDrive/all_data_updated.csv')
df = df.drop(['wind speed', 'device', 'hive number', 'wind speed', 'gust speed', 'weatherID', 'frames', 'cloud coverage', 'rain', 'lat', 'long', 'file name', 'queen acceptance', 'target', 'time', 'date', 'file name'], axis=1)

print("Data Overview:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())

num_features = len(df.columns) - 1
num_rows = (num_features // 3) + (num_features % 3 > 0)

plt.figure(figsize=(15, 5 * num_rows))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(num_rows, 3, i)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

#Preprocessing
df = df.dropna()

Q1 = df['hive temp'].quantile(0.25)
Q3 = df['hive temp'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['hive temp'] < (Q1 - 1.5 * IQR)) | (df['hive temp'] > (Q3 + 1.5 * IQR)))]

Q1 = df['hive pressure'].quantile(0.25)
Q3 = df['hive pressure'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['hive pressure'] < (Q1 - 1.5 * IQR)) | (df['hive pressure'] > (Q3 + 1.5 * IQR)))]

Q1 = df['weather temp'].quantile(0.25)
Q3 = df['weather temp'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['weather temp'] < (Q1 - 1.5 * IQR)) | (df['weather temp'] > (Q3 + 1.5 * IQR)))]

Q1 = df['weather pressure'].quantile(0.25)
Q3 = df['weather pressure'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['weather pressure'] < (Q1 - 1.5 * IQR)) | (df['weather pressure'] > (Q3 + 1.5 * IQR)))]

df.to_csv('cleaned_dataset.csv', index=False)

#EDA after preprocessing
df = pd.read_csv('/content/cleaned_dataset.csv')

print("Data Overview:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())

num_features = len(df.columns) - 1
num_rows = (num_features // 3) + (num_features % 3 > 0)

plt.figure(figsize=(15, 5 * num_rows))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(num_rows, 3, i)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

#attributes @ queen presence 1
df = pd.read_csv('/content/cleaned_dataset.csv')
print("\t\t\t\t\t\t\t\t\t\t Weather Temperature Analysis (Queen Presence = 1)")

temp_counts = {}
min_temp = float('inf')
max_temp = float('-inf')

if 'queen presence' in df.columns and 'weather temp' in df.columns:

    df_filtered = df[(df['queen presence'] == 1) & (df['weather temp'].notnull())]

    for index, row in df_filtered.iterrows():
        temp_value = row['weather temp']

        min_temp = min(min_temp, temp_value)
        max_temp = max(max_temp, temp_value)

        temp_counts[temp_value] = temp_counts.get(temp_value, 0) + 1

    if temp_counts:

        sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)

        max_freq_range = None
        max_freq_count = 0

        temp_ranges = []
        temp_counts_list = []

        for i in range(len(sorted_temps) - 1):
            current_range = (sorted_temps[i][0], sorted_temps[i + 1][0])
            current_count = sum(count for temp, count in sorted_temps if current_range[0] <= temp < current_range[1])

            temp_ranges.append(current_range)
            temp_counts_list.append(current_count)

            if current_count > max_freq_count:
                max_freq_count = current_count
                max_freq_range = current_range

        print(f"Minimum Temperature: {min_temp}")
        print(f"Maximum Temperature: {max_temp}")
        print(f"Range with Most Files: {max_freq_range} (Count: {max_freq_count})")

        intervals = [(15, 20), (20.1, 25), (25.1, 30), (30.1, 35), (35.1, 40), (40.1, 45), (45.1, 50)]
        interval_labels = [f"{start}-{end}" for start, end in intervals]
        interval_counts = [0] * len(intervals)

        for temp, count in temp_counts.items():
            for i, (start, end) in enumerate(intervals):
                if start <= temp < end:
                    interval_counts[i] += count
                    break

        plt.figure(figsize=(10, 6))
        plt.bar(interval_labels, interval_counts)
        plt.xlabel('Temperature Range (°C)')
        plt.ylabel('Frequency Count')
        plt.title('Grouped Temperature Range Distribution (Queen Presence = 1)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No temperature values available for Queen Presence = 1.")
else:
    print("The 'queen presence' or 'hive temp' column is not found in the CSV file.")

print("\t\t\t\t\t\t\t\t\t\t Weather humidity Analysis (Queen Presence = 1)")

temp_counts = {}
min_temp = float('inf')
max_temp = float('-inf')

if 'queen presence' in df.columns and 'weather humidity' in df.columns:

    df_filtered = df[(df['queen presence'] == 1) & (df['weather humidity'].notnull())]

    for index, row in df_filtered.iterrows():
        temp_value = row['weather humidity']

        min_temp = min(min_temp, temp_value)
        max_temp = max(max_temp, temp_value)

        temp_counts[temp_value] = temp_counts.get(temp_value, 0) + 1

    if temp_counts:

        sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)

        max_freq_range = None
        max_freq_count = 0

        temp_ranges = []
        temp_counts_list = []

        for i in range(len(sorted_temps) - 1):
            current_range = (sorted_temps[i][0], sorted_temps[i + 1][0])
            current_count = sum(count for temp, count in sorted_temps if current_range[0] <= temp < current_range[1])

            temp_ranges.append(current_range)
            temp_counts_list.append(current_count)

            if current_count > max_freq_count:
                max_freq_count = current_count
                max_freq_range = current_range

        print(f"Minimum humidity: {min_temp}")
        print(f"Maximum humidity: {max_temp}")
        print(f"Range with Most Files: {max_freq_range} (Count: {max_freq_count})")

        intervals = [(30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]
        interval_labels = [f"{start}-{end}" for start, end in intervals]
        interval_counts = [0] * len(intervals)

        for temp, count in temp_counts.items():
            for i, (start, end) in enumerate(intervals):
                if start <= temp < end:
                    interval_counts[i] += count
                    break

        plt.figure(figsize=(10, 6))
        plt.bar(interval_labels, interval_counts)
        plt.xlabel('humidity Range (°C)')
        plt.ylabel('Frequency Count')
        plt.title('Grouped humidity Range Distribution (Queen Presence = 1)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No humidity values available for Queen Presence = 1.")
else:
    print("The 'queen presence' or 'hive temp' column is not found in the CSV file.")

print("\t\t\t\t\t\t\t\t\t\t Weather pressure Analysis (Queen Presence = 1)")

temp_counts = {}
min_temp = float('inf')
max_temp = float('-inf')

if 'queen presence' in df.columns and 'weather pressure' in df.columns:

    df_filtered = df[(df['queen presence'] == 1) & (df['weather pressure'].notnull())]

    for index, row in df_filtered.iterrows():
        temp_value = row['weather pressure']

        min_temp = min(min_temp, temp_value)
        max_temp = max(max_temp, temp_value)

        temp_counts[temp_value] = temp_counts.get(temp_value, 0) + 1

    if temp_counts:

        sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)

        max_freq_range = None
        max_freq_count = 0

        temp_ranges = []
        temp_counts_list = []

        for i in range(len(sorted_temps) - 1):
            current_range = (sorted_temps[i][0], sorted_temps[i + 1][0])
            current_count = sum(count for temp, count in sorted_temps if current_range[0] <= temp < current_range[1])

            temp_ranges.append(current_range)
            temp_counts_list.append(current_count)

            if current_count > max_freq_count:
                max_freq_count = current_count
                max_freq_range = current_range

        print(f"Minimum pressure: {min_temp}")
        print(f"Maximum pressure: {max_temp}")
        print(f"Range with Most Files: {max_freq_range} (Count: {max_freq_count})")

        intervals = [(1009, 1012), (1012, 1015), (1015, 1018), (1018, 1021)]
        interval_labels = [f"{start}-{end}" for start, end in intervals]
        interval_counts = [0] * len(intervals)

        for temp, count in temp_counts.items():
            for i, (start, end) in enumerate(intervals):
                if start <= temp < end:
                    interval_counts[i] += count
                    break

        plt.figure(figsize=(10, 6))
        plt.bar(interval_labels, interval_counts)
        plt.xlabel('pressure Range (°C)')
        plt.ylabel('Frequency Count')
        plt.title('Grouped pressure Range Distribution (Queen Presence = 1)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No pressure values available for Queen Presence = 1.")
else:
    print("The 'queen presence' or 'hive temp' column is not found in the CSV file.")
print("\t\t\t\t\t\t\t\t\t\t Hive pressure Analysis (Queen Presence = 1)")

temp_counts = {}
min_temp = float('inf')
max_temp = float('-inf')

if 'queen presence' in df.columns and 'hive pressure' in df.columns:

    df_filtered = df[(df['queen presence'] == 1) & (df['hive pressure'].notnull())]

    for index, row in df_filtered.iterrows():
        temp_value = row['hive pressure']

        min_temp = min(min_temp, temp_value)
        max_temp = max(max_temp, temp_value)

        temp_counts[temp_value] = temp_counts.get(temp_value, 0) + 1

    if temp_counts:

        sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)

        max_freq_range = None
        max_freq_count = 0

        temp_ranges = []
        temp_counts_list = []

        for i in range(len(sorted_temps) - 1):
            current_range = (sorted_temps[i][0], sorted_temps[i + 1][0])
            current_count = sum(count for temp, count in sorted_temps if current_range[0] <= temp < current_range[1])

            temp_ranges.append(current_range)
            temp_counts_list.append(current_count)

            if current_count > max_freq_count:
                max_freq_count = current_count
                max_freq_range = current_range

        print(f"Minimum pressure: {min_temp}")
        print(f"Maximum pressure: {max_temp}")
        print(f"Range with Most Files: {max_freq_range} (Count: {max_freq_count})")

        intervals = [(1003, 1006), (1006, 1009), (1009, 1012), (1012, 1015)]
        interval_labels = [f"{start}-{end}" for start, end in intervals]
        interval_counts = [0] * len(intervals)

        for temp, count in temp_counts.items():
            for i, (start, end) in enumerate(intervals):
                if start <= temp < end:
                    interval_counts[i] += count
                    break

        plt.figure(figsize=(10, 6))
        plt.bar(interval_labels, interval_counts)
        plt.xlabel('pressure Range (°C)')
        plt.ylabel('Frequency Count')
        plt.title('Grouped pressure Range Distribution (Queen Presence = 1)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No pressure values available for Queen Presence = 1.")
else:
    print("The 'queen presence' or 'hive temp' column is not found in the CSV file.")
print("\t\t\t\t\t\t\t\t\t\t hive humidity Analysis (Queen Presence = 1)")

temp_counts = {}
min_temp = float('inf')
max_temp = float('-inf')

if 'queen presence' in df.columns and 'hive humidity' in df.columns:

    df_filtered = df[(df['queen presence'] == 1) & (df['hive humidity'].notnull())]

    for index, row in df_filtered.iterrows():
        temp_value = row['hive humidity']

        min_temp = min(min_temp, temp_value)
        max_temp = max(max_temp, temp_value)

        temp_counts[temp_value] = temp_counts.get(temp_value, 0) + 1

    if temp_counts:

        sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)

        max_freq_range = None
        max_freq_count = 0

        temp_ranges = []
        temp_counts_list = []

        for i in range(len(sorted_temps) - 1):
            current_range = (sorted_temps[i][0], sorted_temps[i + 1][0])
            current_count = sum(count for temp, count in sorted_temps if current_range[0] <= temp < current_range[1])

            temp_ranges.append(current_range)
            temp_counts_list.append(current_count)

            if current_count > max_freq_count:
                max_freq_count = current_count
                max_freq_range = current_range

        print(f"Minimum humidity: {min_temp}")
        print(f"Maximum humidity: {max_temp}")
        print(f"Range with Most Files: {max_freq_range} (Count: {max_freq_count})")

        intervals = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]
        interval_labels = [f"{start}-{end}" for start, end in intervals]
        interval_counts = [0] * len(intervals)

        for temp, count in temp_counts.items():
            for i, (start, end) in enumerate(intervals):
                if start <= temp < end:
                    interval_counts[i] += count
                    break

        plt.figure(figsize=(10, 6))
        plt.bar(interval_labels, interval_counts)
        plt.xlabel('humidity Range (°C)')
        plt.ylabel('Frequency Count')
        plt.title('Grouped humidity Range Distribution (Queen Presence = 1)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No humidity values available for Queen Presence = 1.")
else:
    print("The 'queen presence' or 'hive temp' column is not found in the CSV file.")
print("\t\t\t\t\t\t\t\t\t\t Hive Temperature Analysis (Queen Presence = 1)")

temp_counts = {}
min_temp = float('inf')
max_temp = float('-inf')

if 'queen presence' in df.columns and 'hive temp' in df.columns:

    df_filtered = df[(df['queen presence'] == 1) & (df['hive temp'].notnull())]

    for index, row in df_filtered.iterrows():
        temp_value = row['hive temp']

        min_temp = min(min_temp, temp_value)
        max_temp = max(max_temp, temp_value)

        temp_counts[temp_value] = temp_counts.get(temp_value, 0) + 1

    if temp_counts:

        sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)

        max_freq_range = None
        max_freq_count = 0

        temp_ranges = []
        temp_counts_list = []

        for i in range(len(sorted_temps) - 1):
            current_range = (sorted_temps[i][0], sorted_temps[i + 1][0])
            current_count = sum(count for temp, count in sorted_temps if current_range[0] <= temp < current_range[1])

            temp_ranges.append(current_range)
            temp_counts_list.append(current_count)

            if current_count > max_freq_count:
                max_freq_count = current_count
                max_freq_range = current_range

        print(f"Minimum Temperature: {min_temp}")
        print(f"Maximum Temperature: {max_temp}")
        print(f"Range with Most Files: {max_freq_range} (Count: {max_freq_count})")

        intervals = [(15, 20), (20.1, 25), (25.1, 30), (30.1, 35), (35.1, 40), (40.1, 45), (45.1, 50)]
        interval_labels = [f"{start}-{end}" for start, end in intervals]
        interval_counts = [0] * len(intervals)

        for temp, count in temp_counts.items():
            for i, (start, end) in enumerate(intervals):
                if start <= temp < end:
                    interval_counts[i] += count
                    break

        plt.figure(figsize=(10, 6))
        plt.bar(interval_labels, interval_counts)
        plt.xlabel('Temperature Range (°C)')
        plt.ylabel('Frequency Count')
        plt.title('Grouped Temperature Range Distribution (Queen Presence = 1)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No temperature values available for Queen Presence = 1.")
else:
    print("The 'queen presence' or 'hive temp' column is not found in the CSV file.")

#coorelation @ queen presence 1
df = pd.read_csv('/content/cleaned_dataset.csv')
def calculate_correlation(df, cols):
    correlation = df[cols].corr()
    return correlation

def plot_heatmap(ax, correlation_matrix, title):
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

cols_all = ['hive temp', 'hive pressure', 'hive humidity', 'weather temp', 'weather pressure', 'weather humidity']

if all(col in df.columns for col in cols_all):
    df_filtered = df[df['queen presence'] == 1]

    if not df_filtered.empty:
        correlation_all = calculate_correlation(df_filtered, cols_all)

        fig, ax = plt.subplots(figsize=(5, 5))
        plot_heatmap(ax, correlation_all, 'Correlation Heatmap (Queen Presence = 1)')
        plt.tight_layout()
        plt.show()
    else:
        print("No data available for Queen Presence = 1.")
else:
    missing_cols = [col for col in cols_all if col not in df.columns]
    print(f"Missing columns in DataFrame: {missing_cols}")

#attributes @ queen presence 0
df = pd.read_csv('/content/cleaned_dataset.csv')
print("\t\t\t\t\t\t\t\t\t\t Weather Temperature Analysis (Queen Presence = 0)")

temp_counts = {}
min_temp = float('inf')
max_temp = float('-inf')

if 'queen presence' in df.columns and 'weather temp' in df.columns:

    df_filtered = df[(df['queen presence'] == 0) & (df['weather temp'].notnull())]

    for index, row in df_filtered.iterrows():
        temp_value = row['weather temp']

        min_temp = min(min_temp, temp_value)
        max_temp = max(max_temp, temp_value)

        temp_counts[temp_value] = temp_counts.get(temp_value, 0) + 1

    if temp_counts:

        sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)

        max_freq_range = None
        max_freq_count = 0

        temp_ranges = []
        temp_counts_list = []

        for i in range(len(sorted_temps) - 1):
            current_range = (sorted_temps[i][0], sorted_temps[i + 1][0])
            current_count = sum(count for temp, count in sorted_temps if current_range[0] <= temp < current_range[1])

            temp_ranges.append(current_range)
            temp_counts_list.append(current_count)

            if current_count > max_freq_count:
                max_freq_count = current_count
                max_freq_range = current_range

        print(f"Minimum Temperature: {min_temp}")
        print(f"Maximum Temperature: {max_temp}")
        print(f"Range with Most Files: {max_freq_range} (Count: {max_freq_count})")

        intervals = [(15, 20), (20.1, 25), (25.1, 30)]
        interval_labels = [f"{start}-{end}" for start, end in intervals]
        interval_counts = [0] * len(intervals)

        for temp, count in temp_counts.items():
            for i, (start, end) in enumerate(intervals):
                if start <= temp < end:
                    interval_counts[i] += count
                    break

        plt.figure(figsize=(10, 6))
        plt.bar(interval_labels, interval_counts)
        plt.xlabel('Temperature Range (°C)')
        plt.ylabel('Frequency Count')
        plt.title('Grouped Temperature Range Distribution (Queen Presence = 0)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No temperature values available for Queen Presence = 1.")
else:
    print("The 'queen presence' or 'hive temp' column is not found in the CSV file.")


print("\t\t\t\t\t\t\t\t\t\t Weather humidity Analysis (Queen Presence = 1)")

temp_counts = {}
min_temp = float('inf')
max_temp = float('-inf')

if 'queen presence' in df.columns and 'weather humidity' in df.columns:

    df_filtered = df[(df['queen presence'] == 0) & (df['weather humidity'].notnull())]

    for index, row in df_filtered.iterrows():
        temp_value = row['weather humidity']

        min_temp = min(min_temp, temp_value)
        max_temp = max(max_temp, temp_value)

        temp_counts[temp_value] = temp_counts.get(temp_value, 0) + 1

    if temp_counts:

        sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)

        max_freq_range = None
        max_freq_count = 0

        temp_ranges = []
        temp_counts_list = []

        for i in range(len(sorted_temps) - 1):
            current_range = (sorted_temps[i][0], sorted_temps[i + 1][0])
            current_count = sum(count for temp, count in sorted_temps if current_range[0] <= temp < current_range[1])

            temp_ranges.append(current_range)
            temp_counts_list.append(current_count)

            if current_count > max_freq_count:
                max_freq_count = current_count
                max_freq_range = current_range

        print(f"Minimum humidity: {min_temp}")
        print(f"Maximum humidity: {max_temp}")
        print(f"Range with Most Files: {max_freq_range} (Count: {max_freq_count})")

        intervals = [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]
        interval_labels = [f"{start}-{end}" for start, end in intervals]
        interval_counts = [0] * len(intervals)

        for temp, count in temp_counts.items():
            for i, (start, end) in enumerate(intervals):
                if start <= temp < end:
                    interval_counts[i] += count
                    break

        plt.figure(figsize=(10, 6))
        plt.bar(interval_labels, interval_counts)
        plt.xlabel('humidity Range (°C)')
        plt.ylabel('Frequency Count')
        plt.title('Grouped humidity Range Distribution (Queen Presence = 0)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No humidity values available for Queen Presence = 1.")
else:
    print("The 'queen presence' or 'hive temp' column is not found in the CSV file.")


print("\t\t\t\t\t\t\t\t\t\t Weather pressure Analysis (Queen Presence = 1)")

temp_counts = {}
min_temp = float('inf')
max_temp = float('-inf')

if 'queen presence' in df.columns and 'weather pressure' in df.columns:

    df_filtered = df[(df['queen presence'] == 0) & (df['weather pressure'].notnull())]

    for index, row in df_filtered.iterrows():
        temp_value = row['weather pressure']

        min_temp = min(min_temp, temp_value)
        max_temp = max(max_temp, temp_value)

        temp_counts[temp_value] = temp_counts.get(temp_value, 0) + 1

    if temp_counts:

        sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)

        max_freq_range = None
        max_freq_count = 0

        temp_ranges = []
        temp_counts_list = []

        for i in range(len(sorted_temps) - 1):
            current_range = (sorted_temps[i][0], sorted_temps[i + 1][0])
            current_count = sum(count for temp, count in sorted_temps if current_range[0] <= temp < current_range[1])

            temp_ranges.append(current_range)
            temp_counts_list.append(current_count)

            if current_count > max_freq_count:
                max_freq_count = current_count
                max_freq_range = current_range

        print(f"Minimum pressure: {min_temp}")
        print(f"Maximum pressure: {max_temp}")
        print(f"Range with Most Files: {max_freq_range} (Count: {max_freq_count})")

        intervals = [(1009, 1012), (1012, 1015), (1015, 1018), (1018, 1021)]
        interval_labels = [f"{start}-{end}" for start, end in intervals]
        interval_counts = [0] * len(intervals)

        for temp, count in temp_counts.items():
            for i, (start, end) in enumerate(intervals):
                if start <= temp < end:
                    interval_counts[i] += count
                    break

        plt.figure(figsize=(10, 6))
        plt.bar(interval_labels, interval_counts)
        plt.xlabel('pressure Range (°C)')
        plt.ylabel('Frequency Count')
        plt.title('Grouped pressure Range Distribution (Queen Presence = 0)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No pressure values available for Queen Presence = 0.")
else:
    print("The 'queen presence' or 'hive temp' column is not found in the CSV file.")



print("\t\t\t\t\t\t\t\t\t\t Hive pressure Analysis (Queen Presence = 1)")

temp_counts = {}
min_temp = float('inf')
max_temp = float('-inf')

if 'queen presence' in df.columns and 'hive pressure' in df.columns:

    df_filtered = df[(df['queen presence'] == 0) & (df['hive pressure'].notnull())]

    for index, row in df_filtered.iterrows():
        temp_value = row['hive pressure']

        min_temp = min(min_temp, temp_value)
        max_temp = max(max_temp, temp_value)

        temp_counts[temp_value] = temp_counts.get(temp_value, 0) + 1

    if temp_counts:

        sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)

        max_freq_range = None
        max_freq_count = 0

        temp_ranges = []
        temp_counts_list = []

        for i in range(len(sorted_temps) - 1):
            current_range = (sorted_temps[i][0], sorted_temps[i + 1][0])
            current_count = sum(count for temp, count in sorted_temps if current_range[0] <= temp < current_range[1])

            temp_ranges.append(current_range)
            temp_counts_list.append(current_count)

            if current_count > max_freq_count:
                max_freq_count = current_count
                max_freq_range = current_range

        print(f"Minimum pressure: {min_temp}")
        print(f"Maximum pressure: {max_temp}")
        print(f"Range with Most Files: {max_freq_range} (Count: {max_freq_count})")

        intervals = [(1003, 1006), (1006, 1009), (1009, 1012), (1012, 1015)]
        interval_labels = [f"{start}-{end}" for start, end in intervals]
        interval_counts = [0] * len(intervals)

        for temp, count in temp_counts.items():
            for i, (start, end) in enumerate(intervals):
                if start <= temp < end:
                    interval_counts[i] += count
                    break

        plt.figure(figsize=(10, 6))
        plt.bar(interval_labels, interval_counts)
        plt.xlabel('pressure Range (°C)')
        plt.ylabel('Frequency Count')
        plt.title('Grouped pressure Range Distribution (Queen Presence = 0)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No pressure values available for Queen Presence = 1.")
else:
    print("The 'queen presence' or 'hive temp' column is not found in the CSV file.")


print("\t\t\t\t\t\t\t\t\t\t hive humidity Analysis (Queen Presence = 1)")

temp_counts = {}
min_temp = float('inf')
max_temp = float('-inf')

if 'queen presence' in df.columns and 'hive humidity' in df.columns:

    df_filtered = df[(df['queen presence'] == 0) & (df['hive humidity'].notnull())]

    for index, row in df_filtered.iterrows():
        temp_value = row['hive humidity']

        min_temp = min(min_temp, temp_value)
        max_temp = max(max_temp, temp_value)

        temp_counts[temp_value] = temp_counts.get(temp_value, 0) + 1

    if temp_counts:

        sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)

        max_freq_range = None
        max_freq_count = 0

        temp_ranges = []
        temp_counts_list = []

        for i in range(len(sorted_temps) - 1):
            current_range = (sorted_temps[i][0], sorted_temps[i + 1][0])
            current_count = sum(count for temp, count in sorted_temps if current_range[0] <= temp < current_range[1])

            temp_ranges.append(current_range)
            temp_counts_list.append(current_count)

            if current_count > max_freq_count:
                max_freq_count = current_count
                max_freq_range = current_range

        print(f"Minimum humidity: {min_temp}")
        print(f"Maximum humidity: {max_temp}")
        print(f"Range with Most Files: {max_freq_range} (Count: {max_freq_count})")

        intervals = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]
        interval_labels = [f"{start}-{end}" for start, end in intervals]
        interval_counts = [0] * len(intervals)

        for temp, count in temp_counts.items():
            for i, (start, end) in enumerate(intervals):
                if start <= temp < end:
                    interval_counts[i] += count
                    break

        plt.figure(figsize=(10, 6))
        plt.bar(interval_labels, interval_counts)
        plt.xlabel('humidity Range (°C)')
        plt.ylabel('Frequency Count')
        plt.title('Grouped humidity Range Distribution (Queen Presence = 0)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No humidity values available for Queen Presence = 1.")
else:
    print("The 'queen presence' or 'hive temp' column is not found in the CSV file.")


print("\t\t\t\t\t\t\t\t\t\t Hive Temperature Analysis (Queen Presence = 1)")

temp_counts = {}
min_temp = float('inf')
max_temp = float('-inf')

if 'queen presence' in df.columns and 'hive temp' in df.columns:

    df_filtered = df[(df['queen presence'] == 0) & (df['hive temp'].notnull())]

    for index, row in df_filtered.iterrows():
        temp_value = row['hive temp']

        min_temp = min(min_temp, temp_value)
        max_temp = max(max_temp, temp_value)

        temp_counts[temp_value] = temp_counts.get(temp_value, 0) + 1

    if temp_counts:

        sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)

        max_freq_range = None
        max_freq_count = 0

        temp_ranges = []
        temp_counts_list = []

        for i in range(len(sorted_temps) - 1):
            current_range = (sorted_temps[i][0], sorted_temps[i + 1][0])
            current_count = sum(count for temp, count in sorted_temps if current_range[0] <= temp < current_range[1])

            temp_ranges.append(current_range)
            temp_counts_list.append(current_count)

            if current_count > max_freq_count:
                max_freq_count = current_count
                max_freq_range = current_range

        print(f"Minimum Temperature: {min_temp}")
        print(f"Maximum Temperature: {max_temp}")
        print(f"Range with Most Files: {max_freq_range} (Count: {max_freq_count})")

        intervals = [(15, 20), (20.1, 25), (25.1, 30), (30.1, 35), (35.1, 40), (40.1, 45), (45.1, 50)]
        interval_labels = [f"{start}-{end}" for start, end in intervals]
        interval_counts = [0] * len(intervals)

        for temp, count in temp_counts.items():
            for i, (start, end) in enumerate(intervals):
                if start <= temp < end:
                    interval_counts[i] += count
                    break

        plt.figure(figsize=(10, 6))
        plt.bar(interval_labels, interval_counts)
        plt.xlabel('Temperature Range (°C)')
        plt.ylabel('Frequency Count')
        plt.title('Grouped Temperature Range Distribution (Queen Presence = 0)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No temperature values available for Queen Presence = 1.")
else:
    print("The 'queen presence' or 'hive temp' column is not found in the CSV file.")

correlation @ Queen presence 0
df = pd.read_csv('/content/cleaned_dataset.csv')
def calculate_correlation(df, cols):
    correlation = df[cols].corr()
    return correlation

def plot_heatmap(ax, correlation_matrix, title):
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

cols_all = ['hive temp', 'hive pressure', 'hive humidity', 'weather temp', 'weather pressure', 'weather humidity']


if all(col in df.columns for col in cols_all):
    df_filtered = df[df['queen presence'] == 0]
    if not df_filtered.empty:
        correlation_all = calculate_correlation(df_filtered, cols_all)

        fig, ax = plt.subplots(figsize=(5, 5))
        plot_heatmap(ax, correlation_all, 'Correlation Heatmap (Queen Presence = 0)')
        plt.tight_layout()
        plt.show()
    else:
        print("No data available for Queen Presence = 0.")
else:
    missing_cols = [col for col in cols_all if col not in df.columns]
    print(f"Missing columns in DataFrame: {missing_cols}")

#T-test
df_queen_present = df[df['queen presence'] == 1]
features = ['hive temp', 'hive humidity', 'hive pressure', 'weather temp', 'weather pressure', 'weather humidity']
critical_value = 0.05
t_test_results = {}

for feature in features:
    feature_values_present = df_queen_present[feature].dropna()
    feature_values_absent = df[df['queen presence'] == 0][feature].dropna()
    t_stat, p_value = stats.ttest_ind(feature_values_present, feature_values_absent, equal_var=False)
    t_test_results[feature] = {'t-statistic': t_stat}

t_test_results_df = pd.DataFrame(t_test_results).T
print(t_test_results_df)
for feature, result in t_test_results.items():
    if result['t-statistic'] > critical_value:
        print(f"The feature '{feature}' significantly affects queen presence")
    else:
        print(f"The feature '{feature}' does NOT significantly affect queen presence")

#Z-test
df_queen_present = df[df['queen presence'] == 1]
df_queen_absent = df[df['queen presence'] == 0]
features = ['hive temp', 'hive humidity', 'hive pressure', 'weather temp', 'weather pressure', 'weather humidity']
z_test_results = {}
critical_value = 0.05

for feature in features:
    feature_values_present = df_queen_present[feature].dropna()
    feature_values_absent = df_queen_absent[feature].dropna()
    mean_present = feature_values_present.mean()
    mean_absent = feature_values_absent.mean()
    n_present = len(feature_values_present)
    n_absent = len(feature_values_absent)
    std_dev_present = feature_values_present.std()
    std_dev_absent = feature_values_absent.std()
    pooled_std_dev = np.sqrt((std_dev_present ** 2 / n_present) + (std_dev_absent ** 2 / n_absent))
    z_score = (mean_present - mean_absent) / pooled_std_dev
    z_test_results[feature] = {'z-statistic': z_score}

z_test_results_df = pd.DataFrame(z_test_results).T
print(z_test_results_df)

for feature, result in z_test_results.items():
    if (result['z-statistic']) > critical_value:
        print(f"The feature '{feature}' significantly affects queen presence")
    else:
        print(f"The feature '{feature}' does NOT significantly affect queen presence")

#implementing RF model
X = ['hive temp', 'hive humidity', 'hive pressure', 'weather temp', 'weather humidity', 'weather pressure']
X_data = df[X]
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_data)
y = df['queen presence']
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy_rf}, \nPrecision: {precision}, \nRecall: {recall}, \nF1-score: {f1}')

new_data = pd.DataFrame({
    'hive temp': [33.61, 31.19],
    'hive humidity': [23.65, 39.57],
    'hive pressure': [1005.38, 1005.43],
    'weather temp': [30.97, 27.96],
    'weather humidity': [29, 56],
    'weather pressure': [1011, 1012]
})

predictions = rf_model.predict(new_data)
print("Predictions for new data:")
print(predictions)

#hyper parameter tuning
param_grid = {
    'n_estimators': [30, 80, 90, 100, 150],
}
rf_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train, y_train)
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Best Hyperparameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}, \nPrecision: {precision}, \nRecall: {recall}, \nF1-score: {f1}')

feature_importances = pd.Series(best_rf_model.feature_importances_, index=features)
print("\nFeature Importance (Random Forest):")
print(feature_importances.sort_values(ascending=False))

#KNN model
X = pd.DataFrame(X_imputed, columns=['hive temp', 'hive humidity', 'hive pressure', 'weather temp', 'weather humidity', 'weather pressure'])  # Features: hive temp, hive humidity, hive pressure
y = df['queen presence']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy_knn}, \nPrecision: {precision}, \nRecall: {recall}, \nF1-score: {f1}')

new_data = pd.DataFrame({
    'hive temp': [33.61, 31.19],
    'hive humidity': [23.65, 39.57],
    'hive pressure': [1005.38, 1005.43],
    'weather temp': [30.97, 27.96],
    'weather humidity': [29, 56],
    'weather pressure': [1011, 1012]
})

predictions = knn_model.predict(new_data)
print("Predictions for new data:")
print(predictions)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

print("\nFeature Importance (Random Forest):")
print(feature_importances.sort_values(ascending=False))

#hyperparameter tuning
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(df[['hive temp', 'hive humidity', 'hive pressure', 'weather temp', 'weather humidity', 'weather pressure']])

X = pd.DataFrame(X_imputed, columns=['hive temp', 'hive humidity', 'hive pressure', 'weather temp', 'weather humidity', 'weather pressure'])
y = df['queen presence']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_knn = grid_search.best_estimator_

best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Best KNN Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}, \nPrecision: {precision}, \nRecall: {recall}, \nF1-score: {f1}')


rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

print("\nFeature Importance (Random Forest):")
print(feature_importances.sort_values(ascending=False))
new_data = pd.DataFrame({
    'hive temp': [33.61, 31.19],
    'hive humidity': [23.65, 39.57],
    'hive pressure': [1005.38, 1005.43],
    'weather temp': [30.97, 27.96],
    'weather humidity': [29, 56],
    'weather pressure': [1011, 1012]
})

predictions = best_knn.predict(new_data)
print("\nPredictions for new data:")
print(predictions)

#XGBOOST model
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(df[['hive temp', 'hive humidity', 'hive pressure', 'weather temp', 'weather humidity', 'weather pressure']])

X = pd.DataFrame(X_imputed, columns=['hive temp', 'hive humidity', 'hive pressure', 'weather temp', 'weather humidity', 'weather pressure'])
y = df['queen presence']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBClassifier()


xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

accuracy_xgboost = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy_xgboost}, \nPrecision: {precision}, \nRecall: {recall}, \nF1-score: {f1}')

new_data = pd.DataFrame({
    'hive temp': [33.61, 31.19],
    'hive humidity': [23.65, 39.57],
    'hive pressure': [1005.38, 1005.43],
    'weather temp': [30.97, 27.96],
    'weather humidity': [29, 56],
    'weather pressure': [1011, 1012]
})

predictions = xgb_model.predict(new_data)
print("Predictions for new data:")
print(predictions)

feature_importances = xgb_model.feature_importances_
print("Feature Importances:")
for feature_name, importance in zip(X.columns, feature_importances):
    print(f"{feature_name}: {importance}")

#applying JL Lemma
eps = 0.1
n_samples = X_train.shape[0]
original_dim = X_train.shape[1]
reduced_dim = min(johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps), original_dim - 1)
jl_transformer = GaussianRandomProjection(n_components=reduced_dim, random_state=42)
X_train_scaled = jl_transformer.fit_transform(X_train)
X_test_scaled = jl_transformer.transform(X_test)
reduced_dim_actual = X_train_scaled.shape[1]
data_retained_percentage = (reduced_dim_actual / original_dim) * 100
print(f"Original number of features: {original_dim}")
print(f"Reduced number of features: {reduced_dim_actual}")
print(f"Percentage of data retained after scaling: {data_retained_percentage}%")

#implementing RF model on scaled data 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
accuracy_rf_scaled = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy_rf_scaled}, \nPrecision: {precision}, \nRecall: {recall}, \nF1-score: {f1}')
new_data = pd.DataFrame({
    'hive temp': [33.61, 31.19],
    'hive humidity': [23.65, 39.57],
    'hive pressure': [1005.38, 1005.43],
    'weather temp': [30.97, 27.96],
    'weather humidity': [29, 56],
    'weather pressure': [1011, 1012]
})
new_data_scaled = jl_transformer.transform(new_data)
predictions = rf_model.predict(new_data_scaled)
print("Predictions for new data:")
print(predictions)

#KNN model on scaled data 
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred = knn_model.predict(X_test_scaled)
accuracy_knn_scaled = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy_knn_scaled}, \nPrecision: {precision}, \nRecall: {recall}, \nF1-score: {f1}')
new_data_scaled = jl_transformer.transform(new_data)
predictions = knn_model.predict(new_data_scaled)
print("Predictions for new data:")
print(predictions)

#XGBOOST model on scaled data 
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred = xgb_model.predict(X_test_scaled)
accuracy_xgboost_scaled = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy}, \nPrecision: {precision}, \nRecall: {recall}, \nF1-score: {f1}')
new_data_scaled = jl_transformer.transform(new_data)
predictions = knn_model.predict(new_data_scaled)
print("Predictions for new data:")
print(predictions)

#comparison of accuracy on scaled and original dataset
models = ['Random Forest', 'KNN', 'XGBoost']
accuracy_before = [accuracy_rf, accuracy_knn, accuracy_xgboost]
accuracy_after = [accuracy_rf_scaled, accuracy_knn_scaled, accuracy_xgboost_scaled]

x = range(len(models))
fig, ax = plt.subplots(figsize=(8, 8))
width = 0.35
ax.bar(x, accuracy_before, width, label='original', color='skyblue')
ax.bar([p + width for p in x], accuracy_after, width, label='Scaled', color='salmon')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Before and After Scaling')
ax.set_xticks([p + width / 2 for p in x])
ax.set_xticklabels(models)
ax.legend()

plt.show()
