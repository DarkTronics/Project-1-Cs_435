import csv
import math
from matplotlib import pyplot as plt
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.fftpack import rfft  
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model

def dct(scaledData, num, file):
    matrix = scaledData.values
    pi = 3.142857
    dct = []
    m = len(matrix)
    n = len(matrix[0])
    for i in range(m):
        dct.append([None for _ in range(n)])
 
    for i in range(m):
        for j in range(n):
            if (i == 0):
                ci = 1 / (m ** 0.5)
            else:
                ci = (2 / m) ** 0.5
            if (j == 0):
                cj = 1 / (n ** 0.5)
            else:
                cj = (2 / n) ** 0.5
            sum = 0
            for k in range(m):
                for l in range(n):
                    dct1 = matrix[k][l] * math.cos((2 * k + 1) * i * pi / (
                        2 * m)) * math.cos((2 * l + 1) * j * pi / (2 * n))
                    sum = sum + dct1
            dct[i][j] = ci * cj * sum

    df = pd.DataFrame(matrix)
    df = df.iloc[:,:num]
    df.to_csv(file, index=False)
    return df

def pcaFunction(scaledData, num):
    X_meaned = scaledData - np.mean(scaledData , axis = 0)
    cov_mat = np.cov(X_meaned , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:num]
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
    return X_reduced

def process_emg(file):
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

    scalar = StandardScaler() 
    scaled_data = pd.DataFrame(scalar.fit_transform(noOutlierEmg)) # Scaling the data
    pca = PCA()
    pca.fit(scaled_data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) # Calculate the cumulative variances
    n_components = np.argmax(cumulative_variance >= 0.9) + 1
    pc = []
    for i in range(n_components):
        pc.append('PC' + str(i+1))

    pcaEmg = pd.DataFrame(pcaFunction(scaled_data, n_components) , columns = pc) # pca function
    pcaEmg.to_csv('emg_pca.csv', index=False)
    print(pcaEmg)

    autoEncoder('emg_auto.csv', scaled_data,n_components)

    dct(scaled_data, n_components, 'emg_dct.csv')

def process_australian(file):
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

    scalar = StandardScaler() 
    scaled_data = pd.DataFrame(scalar.fit_transform(noOutlierAustralian)) # Scaling the data
    pca = PCA()
    pca.fit(scaled_data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) # Calculate the cumulative variances
    n_components = np.argmax(cumulative_variance >= 0.9) + 1
    pc = []
    for i in range(n_components):
        pc.append('PC' + str(i+1))

    pcaAustralian = pd.DataFrame(pcaFunction(scaled_data, n_components) , columns = pc)
    pcaAustralian.to_csv('australian_pca.csv', index=False)
    print(pcaAustralian)

    autoAustralian = autoEncoder('australian_auto.csv', scaled_data,n_components)

    dctAustralian = dct(scaled_data, n_components, 'australian_dct.csv')

def process_adult(file):
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
    noOutlierAdult['education'] = noOutlierAdult['education'].replace({' Bachelors': 0, ' Some-college': 1, ' Self-emp-inc': 2, ' 11th': 3, ' HS-grad': 4, ' Prof-school': 5, ' Assoc-acdm': 6, ' Assoc-voc': 7, ' 9th': 8, ' 7th-8th': 9, ' 12th':10, ' Masters':11, ' 1st-4th':12, ' 10th':13, ' Doctorate':14, ' 5th-6th':15, ' Preschool':16})
    noOutlierAdult['marital-status'] = noOutlierAdult['marital-status'].replace({' Married-civ-spouse':0, ' Divorced':1, ' Never-married':2, ' Separated':3, ' Widowed': 4, ' Married-spouse-absent':5, ' Married-AF-spouse':6})
    noOutlierAdult['occupation'] = noOutlierAdult['occupation'].replace({' Tech-support':0, ' Craft-repair':1, ' Other-service':2, ' Sales':3, ' Exec-managerial':4, ' Prof-specialty':5, ' Handlers-cleaners':6, ' Machine-op-inspct':7, ' Adm-clerical':8, ' Farming-fishing':9, ' Transport-moving':10, ' Priv-house-serv':11, ' Protective-serv':12, ' Armed-Forces':13})
    noOutlierAdult['relationship'] = noOutlierAdult['relationship'].replace({' Wife':0, ' Own-child':1, ' Husband':2, ' Not-in-family':3, ' Other-relative':4, ' Unmarried':5})
    noOutlierAdult['race'] = noOutlierAdult['race'].replace({' White': 0, ' Asian-Pac-Islander':1, ' Amer-Indian-Eskimo':2, ' Other':3, ' Black':4})
    noOutlierAdult['sex'] = noOutlierAdult['sex'].replace({' Female': 0, ' Male': 1})
    noOutlierAdult['native-country'] = noOutlierAdult['native-country'].replace({' United-States':0, ' Cambodia':1, ' England':2, ' Puerto-Rico':3, ' Canada':4, ' Germany':5, ' Outlying-US(Guam-USVI-etc)':6, ' India':7, ' Japan':8, ' Greece':9, ' South':10, ' China':11, ' Cuba':12, ' Iran':13, ' Honduras':14, ' Philippines':15, ' Italy':16, ' Poland':17, ' Jamaica':18, ' Vietnam':19, ' Mexico':20, ' Portugal':21, ' Ireland':22, ' France':23, ' Dominican-Republic':24, ' Laos':25, ' Ecuador':26, ' Taiwan':27, ' Haiti':28, ' Columbia':29, ' Hungary':30, ' Guatemala':31, ' Nicaragua':32, ' Scotland':33, ' Thailand':34, ' Yugoslavia':35, ' El-Salvador':36, ' Trinadad&Tobago':37, ' Peru':38, ' Hong':39, ' Holand-Netherlands':40})
    noOutlierAdult['income'] = noOutlierAdult['income'].replace({' <=50K':0, ' >50K':1})
    
    noOutlierAdult = noOutlierAdult.drop_duplicates()       
    noOutlierAdult.to_csv('adult.csv', index=False)
    print(noOutlierAdult.describe())
    print(noOutlierAdult)

    scalar = StandardScaler() 
    scaled_data = pd.DataFrame(scalar.fit_transform(noOutlierAdult)) # Scaling the data
    pca = PCA()
    pca.fit(scaled_data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) # Calculate the cumulative variances
    print(cumulative_variance)
    n_components = np.argmax(cumulative_variance >= 0.9) + 1
    pc = []
    for i in range(n_components):
        pc.append('PC' + str(i+1))

    pcaAdult = pd.DataFrame(pcaFunction(scaled_data, n_components) , columns = pc)
    pcaAdult.to_csv('adult_pca.csv', index=False)
    print(pcaAdult)

    autoEncoder('adult_auto.csv', scaled_data,n_components)

    dct(scaled_data, n_components, 'adult_dct.csv')

def autoEncoder(file, data, num):

    # scaler = MinMaxScaler()
    # df_features = scaler.fit_transform(df)

    # # Normalize the test data
    # df_features_test = scaler.transform()  # Adjust based on your needs
    df_features = data
    df_features_test = data

    # Implementation of the Autoencoder Model
    input = Input(shape=(df_features.shape[1],))  # Ensure this shape matches your data
    enc = Dense(15)(input)
    enc = LeakyReLU()(enc)
    enc = Dense(14)(enc)
    enc = LeakyReLU()(enc)

    latent_space = Dense(num, activation="tanh")(enc)

    dec = Dense(32)(latent_space)
    dec = LeakyReLU()(dec)
    dec = Dense(64)(dec)
    dec = LeakyReLU()(dec)
    dec = Dense(units=df_features.shape[1], activation="sigmoid")(dec)

    autoencoder = Model(input, dec)
    autoencoder.compile(optimizer="adam", metrics=["mse"], loss="mse")
    autoencoder.fit(df_features, df_features, epochs=50, batch_size=32, validation_split=0.25)
    encoder = Model(input, latent_space)

    test_au_features = encoder.predict(df_features_test)
    test_au_features_df = pd.DataFrame(test_au_features)
    test_au_features_df.to_csv(file, index=False)
    return test_au_features_df

if __name__=="__main__":
    pd.set_option('future.no_silent_downcasting', True)
    process_emg("emg.txt")
    process_australian('australian.txt')
    process_adult("adult.data")
