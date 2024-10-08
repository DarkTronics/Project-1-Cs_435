import csv
import string
import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

def process_emg(file):
    data_file_name = file
    processed_rows = []
    with open(data_file_name, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        label = int(parts[0])
        features = [0] * 8  # Assuming we have 8 features (1 to 8)
        for part in parts[1:]:
            numcol, value = part.split(':')
            features[int(numcol) - 1] = int(value)  # Inside feature array put the values on the right side of ":"
        processed_rows.append([label] + features)   # Append label to feature

    with open("emg.csv", mode='w', newline='') as file: # Write the format change of emg.txt to emg.csv
        writer = csv.writer(file)
        writer.writerow(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9'])
        writer.writerows(processed_rows)

    originalEmgData = pd.read_csv("emg.csv")
    print(originalEmgData.describe())

    for i in range(1, 9): # For each 8 features, use IQR to remove outliers
        if i == 1:
            q1, q3 = np.percentile(originalEmgData['Feature 1'], [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            noOutlierEmg = originalEmgData[(originalEmgData['Feature 1'] >= lower_bound) & (originalEmgData['Feature 1'] <= upper_bound)]
        else:
            q1, q3 = np.percentile(noOutlierEmg['Feature ' + str(i)], [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            noOutlierEmg = noOutlierEmg[(noOutlierEmg['Feature ' + str(i)] >= lower_bound) & (noOutlierEmg['Feature ' + str(i)] <= upper_bound)]

    print(noOutlierEmg.describe())
    noOutlierEmg.to_csv("emg.csv", index=False)
        
    # fig, axs = plt.subplots(9,1,dpi=95, figsize=(7,17))
    # i = 0
    # for col in originalAustralianData.columns:
    #     axs[i].boxplot(originalAustralianData[col], vert=False)
    #     axs[i].set_ylabel(col)
    #     i+=1
    # fig, axs = plt.subplots(9,1,dpi=95, figsize=(7,17))
    # i = 0
    # for col in originalEmgData.columns:
    #     axs[i].boxplot(originalEmgData[col], vert=False)
    #     axs[i].set_ylabel(col)
    #     i+=1
    # plt.show()

    #Binning
    # for column in noOutlierEmg.columns:
    #     noOutlierEmg[column] = noOutlierEmg[column].sort_values().values

    # noOutlierEmg['Feature 1'] = noOutlierEmg['Feature 1'].sort_values().values
    # noOutlierEmg = noOutlierEmg.sort_values(by=['Feature 1'], ascending=True)
    noOutlierEmg.to_csv('emg.csv', index=False)

    # originalAustralianData['Feature 1 bin'] = pd.qcut(originalAustralianData['Feature 1'], q=200)
    # originalAustralianData.to_csv('emg.csv', index=False)
    print(noOutlierEmg)

def process_australian(file):
    data_file_name = file
    processed_rows = []
    with open(data_file_name, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        label = int(parts[0])
        features = [0] * 14  # Assuming we have 14 features (1 to 14)
        for part in parts[1:]:
            numcol, value = part.split(':')
            features[int(numcol) - 1] = int(float(value))  # Inside feature array put the values on the right side of ":"
        processed_rows.append([label] + features)   # Append label to feature

    with open("australian.csv", mode='w', newline='') as file: # Write the format change of emg.txt to emg.csv
        writer = csv.writer(file)
        writer.writerow(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10', 'Feature 11', 'Feature 12', 'Feature 13', 'Feature 14', 'Feature 15'])
        writer.writerows(processed_rows)

    originalAustralianData = pd.read_csv("australian.csv")
    print(originalAustralianData.describe())

    for i in range(1, 15): # For each 14 features, use IQR to remove outliers
        if i == 1:
            q1, q3 = np.percentile(originalAustralianData['Feature 1'], [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            noOutlierAustralian = originalAustralianData[(originalAustralianData['Feature 1'] >= lower_bound) & (originalAustralianData['Feature 1'] <= upper_bound)]
        else:
            q1, q3 = np.percentile(noOutlierAustralian['Feature ' + str(i)], [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            noOutlierAustralian = noOutlierAustralian[(noOutlierAustralian['Feature ' + str(i)] >= lower_bound) & (noOutlierAustralian['Feature ' + str(i)] <= upper_bound)]

    print(noOutlierAustralian.describe())
    noOutlierAustralian.to_csv("australian.csv", index=False)
        
    # fig, axs = plt.subplots(15,1,dpi=95, figsize=(7,17))
    # i = 0
    # for col in originalAustralianData.columns:
    #     axs[i].boxplot(originalAustralianData[col], vert=False)
    #     axs[i].set_ylabel(col)
    #     i+=1
    # fig, axs = plt.subplots(15,1,dpi=95, figsize=(7,17))
    # i = 0
    # for col in noOutlierAustralian.columns:
    #     axs[i].boxplot(noOutlierAustralian[col], vert=False)
    #     axs[i].set_ylabel(col)
    #     i+=1
    # plt.show()

    # #Binning
    # for column in noOutlierAustralian.columns:
    #     noOutlierAustralian[column] = noOutlierAustralian[column].sort_values().values
    noOutlierAustralian.to_csv('australian.csv', index=False)

    # originalAustralianData['Feature 1 bin'] = pd.qcut(originalAustralianData['Feature 1'], q=200)
    # originalAustralianData.to_csv('emg.csv', index=False)
    print(noOutlierAustralian)

def process_adult(file):
    data_file_name = file
    data_file_name = file
    processed_rows = []
    with open(data_file_name, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split(',')
        label = parts[0]
        features = [0] * 14  # Assuming we have 14 features (1 to 14)
        i = 0
        for part in parts[1:]:
            features[i] = str(part)
            i+=1
        processed_rows.append([label] + features)   # Append label to feature
    with open("adult.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10', 'Feature 11', 'Feature 12', 'Feature 13', 'Feature 14', 'Features 15'])
        writer.writerows(processed_rows)
    
    originalAdultData = pd.read_csv("adult.csv")
    #Dealing with missing values
    noOutlierAdult = originalAdultData[~originalAdultData['Feature 2'].str.contains(r'\?', na=False)]
    medianHoursWorked = noOutlierAdult['Feature 13'].median()
    print(medianHoursWorked)
    for index, row in noOutlierAdult.iterrows(): # replace the missing hour worked values with the median
        if row['Feature 13'] == ' ?':
            noOutlierAdult.at[index, 'Feature 13'] = medianHoursWorked
    noOutlierAdult = noOutlierAdult[~noOutlierAdult['Feature 14'].str.contains(r'\?', na=False)]
    noOutlierAdult = noOutlierAdult[~noOutlierAdult['Feature 7'].str.contains(r'\?', na=False)]
    noOutlierAdult.drop(noOutlierAdult.tail(1).index,inplace=True)
    
    noOutlierAdult.to_csv('adult.csv', index=False)
    print(noOutlierAdult.describe())
    print(noOutlierAdult)

if __name__=="__main__":
    process_emg("emg.txt")
    process_australian('australian.txt')
    process_adult("adult.data")