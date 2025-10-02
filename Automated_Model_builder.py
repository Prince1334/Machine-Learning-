import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
from sklearn.svm import SVR
import os
import torch
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

print("Enter dataset which contain only numeric values, otherwise it generate error and also make sure that values does not contain any special character or spaces between")
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Choose dataset format:")
print("1. CSV")
print("2. Excel")
print("3. Exit")
choose = input("Enter choice: ")
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
if choose == "1":
    path = input("Enter path to CSV file: ")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        print("The specified file does NOT exist.")
        exit()
elif choose == "2":
    path = input("Enter path to Excel file: ")
    if os.path.exists(path):
        df = pd.read_excel(path)
        df.to_csv('data.csv', index=False)
        df = pd.read_csv('data.csv')
    else:
        print("The specified file does NOT exist.")
        exit()
elif choose == "3":
    print("Exiting...")
    exit()
else:
    print("Invalid choice")
    exit()
#----------------------------------------------------------------------------------------------------------------------------------------------------------------#
if df.isnull().sum().sum() == 0:
    print("No missing values detected.")
else:    
    print("Missing values detected.")
    if df.shape[0] > 1000:
        print("Would you like to fill all Rows which have null values, we found that your dataset contain more than 1000 rows, it is not mandatory to fill null values:")
        print("1. Yes")
        print("2. No")
        ch1 = input("Enter Choice: ")
        if ch1 == "1":
            print(" 1.Fill with: Front Fill, it uses previous cell value to fill null value")
            print(" 2.Fill with: Back Fill, it uses next cell value to fill null value")
            choose = input("Enter Choice: ")
        if choose == "1":
            print("1) Mean\n2) Median\n3) Mode")
            method = input("Enter choice: ")
            if method == "1":
                df.fillna(df.mean(),method='ffill', inplace=True)
            elif method == "2":
                df.fillna(df.median(),method='ffill', inplace=True)
            elif method == "3":
                df.fillna(df.mode().iloc[0],method='ffill', inplace=True)
            else:
                print("Invalid choice")
                pass
        if choose == "2":
            print("1) Mean\n2) Median\n3) Mode")
            method1 = input("Enter choice: ")
            if method1 == "1":
                df.fillna(df.mean(),method='bfill', inplace=True)
            elif method1 == "2":
                df.fillna(df.median(),method='bfill', inplace=True)
            elif method1 == "3":
                df.fillna(df.mode().iloc[0],method='bfill', inplace=True)
            elif method1 == "4":
                pass       
        elif ch1 == "2":            
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
    else:
        print("Missing values detected. Choose method to fill, we found that your dataset contain less than 1000 rows, it is mandatory to fill null values otherwise it create problems:")
        ch1 = input("Enter Choice: ")
        if ch1 == "1":
            print(" 1.Fill with: Front Fill, it uses previous cell value to fill null value")
            print(" 2.Fill with: Back Fill, it uses next cell value to fill null value")
            choose = input("Enter Choice: ")
        if choose == "1":
            print("1) Mean\n2) Median\n3) Mode")
            method = input("Enter choice: ")
            if method == "1":
                df.fillna(df.mean(),method='ffill', inplace=True)
            elif method == "2":
                df.fillna(df.median(),method='ffill', inplace=True)
            elif method == "3":
                df.fillna(df.mode().iloc[0],method='ffill', inplace=True)
            else:
                print("Invalid choice")
                pass
        if choose == "2":
            print("1) Mean\n2) Median\n3) Mode")
            method1 = input("Enter choice: ")
            if method1 == "1":
                df.fillna(df.mean(),method='bfill', inplace=True)
            elif method1 == "2":
                df.fillna(df.median(),method='bfill', inplace=True)
            elif method1 == "3":
                df.fillna(df.mode().iloc[0],method='bfill', inplace=True)
            elif method1 == "4":
                pass       
        elif ch1 == "2":            
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------#
print("Would you like to show graphs for data visualization?")
print("1. Yes")
print("2. No")
ch = input("Enter choice:")
if ch == "1":
        while True:
            print("Choose graph type:")
            print("1. Scatter Plot")
            print("2. Histogram")
            print("3. Exit")
            ch = input("Enter choice: ")
            if ch == "1":
                xinp = input("Enter column X name for Scatter Plot (Enter only One column name): ")
                yinp = input("Enter column Y name for Scatter Plot: ")
                plt.scatter(df[xinp], df[yinp])
                plt.xlabel(xinp)
                plt.ylabel(yinp)
                plt.title(f"Scatter Plot of {xinp} vs {yinp}")
                plt.show()
            elif ch == "2":
                xinp = input("Enter column name for Histogram: ")
                plt.hist(df[xinp], bins=10, edgecolor='black')
                plt.xlabel(xinp)
                plt.ylabel("Frequency")
                plt.title(f"Histogram of {xinp}")   
                plt.show()
            elif ch == "3":
                break
else:
    pass 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------#
def RandomForestReg(X_train, y_train): 
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    while True:
        print("1) Predict\n2) Exit")
        ch = input("Enter choice: ")
        if ch == "1":
            print("Training......")
            raw_input = input("Enter comma-separated values: ")
            values = np.array([float(i) for i in raw_input.split(",")]).reshape(1, -1)
            prediction = model.predict(values)
            print("Prediction:", prediction)
        elif ch == "2":
            break  
        
#----------------------------------------------------------------------------------------------------------------------------------------------------------------# 
def kmeans(X_train,y_train):
    model = KMeans()
    model.fit(X_train, y_train)
    while True:
        print("1) Predict\n2) Exit")
        ch = input("Enter choice: ")
        if ch == "1":
            print("Traning......")
            raw_input = input("Enter comma-separated values: ")
            values = np.array([float(i) for i in raw_input.split(",")]).reshape(1, -1)
            prediction = model.predict(values)
            print("Prediction:", prediction)
        elif ch == "2":

            break  
#----------------------------------------------------------------------------------------------------------------------------------------------------------------# 
def trees(X_train,y_train):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    while True:
        print("1) Predict\n2) Exit")
        ch = input("Enter choice: ")
        if ch == "1":
            print("Traning......")
            raw_input = input("Enter comma-separated values: ")
            values = np.array([float(i) for i in raw_input.split(",")]).reshape(1, -1)
            prediction = model.predict(values)
            print("Prediction:", prediction)
        elif ch == "2":
        
            break 
        else:
            break
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def svr(X_train,y_train):
    model = SVR()
    model.fit(X_train, y_train)
    while True:
        print("1) Predict\n2) Exit")
        ch = input("Enter choice: ")
        if ch == "1":
            print("Traning......")
            raw_input = input("Enter comma-separated values: ")
            values = np.array([float(i) for i in raw_input.split(",")]).reshape(1, -1)
            prediction = model.predict(values)
            print("Prediction:", prediction)
        elif ch == "2":
            break 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def gb(X_train,y_train):
    model = GradientBoostingRegressor() 
    model.fit(X_train, y_train)   
    while True:
        print("1) Predict\n2) Exit")
        ch = input("Enter choice: ")
        if ch == "1":
            print("Traning......")
            raw_input = input("Enter comma-separated values: ")
            values = np.array([float(i) for i in raw_input.split(",")]).reshape(1, -1)
            prediction = model.predict(values)
            print("Prediction:", prediction)
        elif ch == "2":
            break 

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
print("-----------------------------------------------------------------------------------------------------------------------------")        
target_col = input("Enter target column name: ")
X = df.drop(columns=[target_col])
y = df[target_col]
X_reg, y_reg = make_regression(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_reg,y_reg, test_size=0.2, random_state=42)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#
print("-----------------------------------------------------------------------------------------------------------------------------")
print("All Below are the best Machine Learning Models for Microorganism Growth Prediction:")
print("Choose model type:")
print("1) Random Forest Regressor")
print("2) K-Means")
print("3) Decision Tree Regressor")
print("4) Support Vector Regression")
print("5) Gradient Boosting Regressor")
print("6) Exit")
model_choice = input("Enter choice: ")

if model_choice == "1":
    RandomForestReg(X_train, y_train)
elif model_choice == "2":
    kmeans(X_train,y_train)
elif model_choice == "3":
    trees(X_train,y_train)
    exit()
elif model_choice == "4":
    svr(X_train,y_train)
elif model_choice == "5":
    gb(X_train,y_train)
elif model_choice == "6":
    print("Exiting...")
    exit()
else:
    print("Invalid model choice")
    exit()
#----------------------------------------------------------------------------------------------------------------------------------------------------------------#             