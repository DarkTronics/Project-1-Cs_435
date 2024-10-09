import csv
import string
import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.fftpack import rfft  
import matplotlib.pyplot as plt
import plotly.express as px

def dft(file, num):
    data_file_name = file
    originalData = pd.read_csv(file)
    n_dimensions = num
    data_dft = rfft(originalData,n=n_dimensions)
    print(data_dft)


def pca(file, num):
    data_file_name = file
    originalData = pd.read_csv(file)

    pcnum = num
    pc = []
    for i in range(num):
        pc.append('PC' + str(i+1))

    scalar = StandardScaler() 
    scaled_data = pd.DataFrame(scalar.fit_transform(originalData)) #scaling the data
    pca = PCA(n_components = num)
    pca.fit(scaled_data)
    data_pca = pca.fit_transform(scaled_data)
    data_pca = pd.DataFrame(data_pca,columns=pc)
    components = pca.fit_transform(scaled_data)
    print(data_pca)

    sns.heatmap(scaled_data.corr())
    plt.show()
    sns.heatmap(data_pca.corr())
    plt.show()

    if data_file_name == 'adult.csv':
        fig = px.scatter(components, x=0, y=1, color=originalData['income'])
        fig.show()
    else:
        fig = px.scatter(components, x=0, y=1, color=originalData['Feature 1'])
        fig.show()

def process_emg(file, numd):
    print("------------------------------------------------------------------------Processing Emg----------------------------------------------------------")
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

    noOutlierEmg = noOutlierEmg.drop_duplicates()       

    noOutlierEmg.to_csv('emg.csv', index=False)
    print(noOutlierEmg.describe())
    print(noOutlierEmg)
    pca('emg.csv', numd)
    dft('emg.csv', numd)

def process_australian(file, numd):
    print("------------------------------------------------------------------------Processing Australian----------------------------------------------------------")
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

    noOutlierAustralian = noOutlierAustralian.drop_duplicates()       
    noOutlierAustralian.to_csv("australian.csv", index=False)

    print(noOutlierAustralian.describe())
    print(noOutlierAustralian)
    pca('australian.csv',numd)

def process_adult(file, numd):
    print("------------------------------------------------------------------------Processing Adult------------------------------------------------------------------------")
    data_file_name = file
    data_file_name = file
    processed_rows = []
    with open(data_file_name, 'r') as file:
        lines = file.readlines()

    # Parse original adult
    for line in lines:
        parts = line.strip().split(',')
        features = [0] * 15  # Assuming we have 14 features (1 to 14)
        i = 0
        for part in parts[0:]:
            features[i] = part
            i+=1
        processed_rows.append(features)   # Append label to feature
    with open("adult.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
                         'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
        writer.writerows(processed_rows)

    #Dealing with missing values
    originalAdultData = pd.read_csv("adult.csv")
    noOutlierAdult = originalAdultData[~originalAdultData['workclass'].str.contains(r'\?', na=False)]
    medianHoursWorked = noOutlierAdult['hours-per-week'].median()
    for index, row in noOutlierAdult.iterrows(): # replace the missing hour worked values with the median
        if row['hours-per-week'] == ' ?':
            noOutlierAdult.at[index, 'hours-per-week'] = medianHoursWorked
    noOutlierAdult = noOutlierAdult[~noOutlierAdult['native-country'].str.contains(r'\?', na=False)]
    noOutlierAdult = noOutlierAdult[~noOutlierAdult['occupation'].str.contains(r'\?', na=False)]
    noOutlierAdult.drop(noOutlierAdult.tail(1).index,inplace=True)

    noOutlierAdult['workclass'] = noOutlierAdult['workclass'].replace({' Private': 0, ' Self-emp-not-inc': 1, ' Self-emp-inc': 2, ' Federal-gov': 3, ' Local-gov': 4, ' State-gov': 5, ' Without-pay': 6, ' Never-worked': 7})
    noOutlierAdult['education'] = noOutlierAdult['education'].replace({' Bachelors': 0, ' Some-college': 1, ' Self-emp-inc': 2, ' 11th': 3, ' HS-grad': 4, ' Prof-school': 5, 
                                                                       ' Assoc-acdm': 6, ' Assoc-voc': 7, ' 9th': 8, ' 7th-8th': 9, ' 12th':10, ' Masters':11, ' 1st-4th':12, ' 10th':13, ' Doctorate':14, ' 5th-6th':15, ' Preschool':16})
    noOutlierAdult['marital-status'] = noOutlierAdult['marital-status'].replace({' Married-civ-spouse':0, ' Divorced':1, ' Never-married':2, ' Separated':3, ' Widowed': 4, ' Married-spouse-absent':5, ' Married-AF-spouse':6})
    noOutlierAdult['occupation'] = noOutlierAdult['occupation'].replace({' Tech-support':0, ' Craft-repair':1, ' Other-service':2, ' Sales':3, ' Exec-managerial':4, ' Prof-specialty':5, 
                                                                         ' Handlers-cleaners':6, ' Machine-op-inspct':7, ' Adm-clerical':8, ' Farming-fishing':9, ' Transport-moving':10, ' Priv-house-serv':11, ' Protective-serv':12, ' Armed-Forces':13})
    noOutlierAdult['relationship'] = noOutlierAdult['relationship'].replace({' Wife':0, ' Own-child':1, ' Husband':2, ' Not-in-family':3, ' Other-relative':4, ' Unmarried':5})
    noOutlierAdult['race'] = noOutlierAdult['race'].replace({' White': 0, ' Asian-Pac-Islander':1, ' Amer-Indian-Eskimo':2, ' Other':3, ' Black':4})
    noOutlierAdult['sex'] = noOutlierAdult['sex'].replace({' Female': 0, ' Male': 1})
    noOutlierAdult['native-country'] = noOutlierAdult['native-country'].replace({' United-States':0, ' Cambodia':1, ' England':2, ' Puerto-Rico':3, ' Canada':4, ' Germany':5, ' Outlying-US(Guam-USVI-etc)':6, ' India':7, 
                                                                                 ' Japan':8, ' Greece':9, ' South':10, ' China':11, ' Cuba':12, ' Iran':13, ' Honduras':14, ' Philippines':15, ' Italy':16, ' Poland':17, 
                                                                                 ' Jamaica':18, ' Vietnam':19, ' Mexico':20, ' Portugal':21, ' Ireland':22, ' France':23, ' Dominican-Republic':24, ' Laos':25, 
                                                                                 ' Ecuador':26, ' Taiwan':27, ' Haiti':28, ' Columbia':29, ' Hungary':30, ' Guatemala':31, ' Nicaragua':32, ' Scotland':33, ' Thailand':34, 
                                                                                 ' Yugoslavia':35, ' El-Salvador':36, ' Trinadad&Tobago':37, ' Peru':38, ' Hong':39, ' Holand-Netherlands':40})
    noOutlierAdult['income'] = noOutlierAdult['income'].replace({' <=50K':0, ' >50K':1})
    
    noOutlierAdult = noOutlierAdult.drop_duplicates()       
    noOutlierAdult.to_csv('adult.csv', index=False)
    print(noOutlierAdult.describe())
    print(noOutlierAdult)
    pca('adult.csv',numd)


if __name__=="__main__":
    pd.set_option('future.no_silent_downcasting', True)
    process_emg("emg.txt", 3)
    # process_australian('australian.txt', 3)
    # process_adult("adult.data", 2)