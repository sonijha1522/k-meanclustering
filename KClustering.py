import numpy as np
import matplotlib
import random as rd
import matplotlib as mp
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv ("Income.csv")
print(df.head())
plt.title('kmean plot')
#plt.xlabel('Age')
#plt.ylabel('Income')
#X = df['Age']
#Y = df['Income']
#Visualise data points
#plt.scatter(X, Y, c = 'black')

plt.show()

#number of clusters = 3
# Select random observation as centroids
K = 3
Centroids = (df.sample(n=K))
plt.scatter(df["Age"],df["Income"],c='black')
plt.scatter(Centroids["Age"],Centroids["Income"],c='red')
plt.xlabel('Age')
plt.ylabel('Income (In Thousands)')
plt.show()

diff = 1
j=0

while(diff!=0):
    XD=df
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Age"]-row_d["Age"])**2
            d2=(row_c["Income"]-row_d["Income"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        df[i]=ED
        i=i+1

    C=[]
    for index,row in df.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    df["Cluster"]=C
    Centroids_new = df.groupby(["Cluster"]).mean()[["Age","Income"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Age'] - Centroids['Age']).sum() + (Centroids_new['Income'] - Centroids['Income']).sum()
        print(diff.sum())
    Centroids = df.groupby(["Cluster"]).mean()[["Age","Income"]]

color=['blue','green','cyan']
for k in range(K):
    data=df[df["Cluster"]==k+1]
    plt.scatter(data["Age"],data["Income"],c=color[k])
plt.scatter(Centroids["Age"],Centroids["Income"],c='red')
plt.xlabel('Age')
plt.ylabel('Income (In Thousands)')
plt.show()