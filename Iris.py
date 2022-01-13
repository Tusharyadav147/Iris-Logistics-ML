import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


#loading dataset
df = pd.read_csv("Iris.csv")
print("\nOur Dataset Is:- \n",df.head(10))
print("\nShape Of Dataset Is = ",df.shape)

#droping unwanted column
df = df.drop(columns="Id")
print("\nAfter Droping 'Id' column Dataset will be:- \n",df.head(10))
print("\nShape after droping a column = ",df.shape)

#change the species column values 
species = {"Iris-setosa": 0, "Iris-versicolor": 1,"Iris-virginica": 2}
df["Species"] = df["Species"].map(species)
print("\nAfter replacing Species values:- ",df.head())

# to generate a profile report of iris dataset
pr = ProfileReport(df)
pr.to_file("Iris_report.html")

# to display stats about data
print("\nStats value of dataset is :-",df.describe())

#some information about dataset
print("\nSome Information about dataset",df.info())

#count how many null values we have
print("\nTotal null values in the dataset",df.isnull().sum())

#separate independent and dependent variable
x = df.drop(columns= "Species")
y = df["Species"]

#making training ans testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)

# preparing model
model = LogisticRegression()
model.fit(x_train, y_train)

#looking at the c and m values
print("\nThe interception is : {}".format(model.intercept_))
print("\nThe coefficient c value is : {}".format(model.coef_))

#making prediction through our model
print("\nAccuray Score of our model is :- {}".format(model.score(x_test, y_test)*100))

#save the model ro disk
model = pickle.dump(model, open("ML_linear_iris.pkl", "wb"))