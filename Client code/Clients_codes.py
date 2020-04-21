import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff

# Importing dataset and examining it
dataset = pd.read_csv("Clients.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())
# get the uniqua values in the columns term and job
print(dataset["term"].unique())
print(dataset["job"].unique())
# change admin. to admin
dataset.loc[dataset["job"]=="admin.","job"]="admin"
#change n to no
dataset.loc[dataset["term"]=="n","term"]="no"
print(dataset["term"].unique())
print(dataset["job"].unique())
# Converting Categorical features into Numerical features
dataset["job"]=dataset["job"].map({"admin":1, "unemployed":2, "management":3, "housemaid":4,"entrepreneur":5, "student":6, "blue-collar":7, "self-employed":8, "retired":9, "technician":10, "services":11})
dataset["marital"]=dataset["marital"].map({"married":2, "divorced":3, "single":1})
dataset["education"]=dataset["education"].map({"secondary":2, "primary":1, "tertiary":3})
dataset["default"]=dataset["default"].map({"yes":1,"no":0})
dataset["housing"]=dataset["housing"].map({"yes":1,"no":0})
dataset["personal"]=dataset["personal"].map({"yes":1,"no":0})
dataset["term"]=dataset["term"].map({"yes":1,"no":0})

print(dataset.head())
print(dataset.dtypes)

#Plotting Correlation Heatmap
# corrs = dataset.corr()
# figure = ff.create_annotated_heatmap(
#     z=corrs.values,
#     x=list(corrs.columns),
#     y=list(corrs.index),
#     annotation_text=corrs.round(2).values,
#     showscale=True)
# offline.plot(figure,filename='Clientcorrheatmap.html')

#from the result of the correlation heatmap, there is no strong correlation among the columns. So no column is dropped.

#Therefore, we will set X to the entire data set

X=dataset

# Defining if Age is Youth or Old
def converterAge(column):
    if column >= 40:
        return 1 # Old
    else:
        return 0 # Youth

X['age'] = X['age'].apply(converterAge)

# Defining if Balance Debit or Credit
def converterBalance(column):
    if column >= 0:
        return 1 # Credit
    else:
        return 0 # Debit

X['balance'] = X['balance'].apply(converterBalance)

print(X.head())
print(X.info())

#We will create 3 subset of the column, based on Personal and Money based variables

#dividing data into subsets

#Personal data

subset1 = X[['job','marital','education']]

#Financial Data 

subset2 = X[['default','housing','personal', 'term','age']]

#all Data

subset3 = X

# Normalizing numerical features so that each feature has mean 0 and variance 1

feature_scaler = StandardScaler()
X1 = feature_scaler.fit_transform(subset1)
X2 = feature_scaler.fit_transform(subset2)
X3= feature_scaler.fit_transform(subset3)

# Analysis on subset1 - Personal Data
# Finding the number of clusters (K) - Elbow Plot Method
# inertia = []
# for i in range(1,11):
#     kmeans = KMeans(n_clusters = i, random_state = 100)
#     kmeans.fit(X1)
#     inertia.append(kmeans.inertia_)

# plt.plot(range(1, 11), inertia)
# plt.title('The Elbow Plot')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()

# # # Running KMeans to generate labels
# kmeans = KMeans(n_clusters = 2)
# kmeans.fit(X1)

# # Implementing t-SNE to visualize dataset
# tsne = TSNE(n_components = 2, perplexity =50,n_iter=500)
# x_tsne = tsne.fit_transform(X1)

# #age = list(X['age'])
# job = list(X['job'])
# marital = list(X['marital'])
# education = list(X['education'])
# #balance = list(X['balance'])
# #personal = list(X['personal'])


# data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
#                     marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
#                                 text=[f'Job: {a}; MaritalStatus:{c};Education:{d}' for a,c,d in list(zip(job,marital,education))],
#                                 hoverinfo='text')]

# layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
#                     xaxis = dict(title='First Dimension'),
#                     yaxis = dict(title='Second Dimension'))
# fig = go.Figure(data=data, layout=layout)
# offline.plot(fig,filename='ex1-subset-1-50p-500i-c2.html')

# # # Analysis on subset2 - Financial Data
# # Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X2)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# # Running KMeans to generate labels
kmeans = KMeans(n_clusters = 7)
kmeans.fit(X2)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =100,n_iter=500)
x_tsne = tsne.fit_transform(X2)
default = list(X['default'])
balance = list(X['balance'])
housing = list(X['housing'])
personal = list(X['personal'])
term = list(X['term'])
age = list(X['age'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Default: {a}; Balance: {b}; Housing:{c}; Personal:{d}; Term:{e}:Age:{f}' for a,b,c,d,e,f in list(zip(default,balance,housing,personal,term,age))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='ex1-subset-2-50p-500i-c4.html')

# # Analysis on subset3 - All Data
# # Finding the number of clusters (K) - Elbow Plot Method
# inertia = []
# for i in range(1,11):
#     kmeans = KMeans(n_clusters = i, random_state = 100)
#     kmeans.fit(X3)
#     inertia.append(kmeans.inertia_)

# plt.plot(range(1, 11), inertia)
# plt.title('The Elbow Plot')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()

# # # Running KMeans to generate labels
# kmeans = KMeans(n_clusters = 4)
# kmeans.fit(X3)

# # Implementing t-SNE to visualize dataset
# tsne = TSNE(n_components = 2, perplexity =50,n_iter=500)
# x_tsne = tsne.fit_transform(X3)

# age = list(X['age'])
# job = list(X['job'])
# marital = list(X['marital'])
# education = list(X['education'])
# default = list(X['default'])
# balance = list(X['balance'])
# housing = list(X['housing'])
# personal = list(X['personal'])
# term = list(X['term'])

# data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
#                     marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
#                                 text=[f'{a}; Job: {b}; MaritalStatus:{c}, Education:{d},Default: {e}; Balance: {f}; Housing:{g}, Personal:{h}, Term:{i}' for a,b,c,d,e,f,g,h,i in list(zip(age,job,marital,education,default,balance,housing,personal,term))],
#                                 hoverinfo='text')]

# layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
#                     xaxis = dict(title='First Dimension'),
#                     yaxis = dict(title='Second Dimension'))
# fig = go.Figure(data=data, layout=layout)
# offline.plot(fig,filename='ex1-subset-3-50p-500i-c8.html')


