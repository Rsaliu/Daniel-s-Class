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
dataset = pd.read_csv("Patches.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Converting Categorical features into Numerical features
dataset['Tree'] = dataset['Tree'].map({'Spruce':1, 'Other':0})

print(dataset.head())
print(dataset.info())

# Plotting Correlation Heatmap
# corrs = dataset.corr()
# figure = ff.create_annotated_heatmap(
#     z=corrs.values,
#     x=list(corrs.columns),
#     y=list(corrs.index),
#     annotation_text=corrs.round(2).values,
#     showscale=True)
# offline.plot(figure,filename='corrheatmap.html')


#from the result of the correlation heatmap, there is no correlation among the columns. So no column is dropped.

#Therefore, we will set X to the entire data set

X=dataset

# Defining if Elevation w high or low
def converterElevation(column):
    if column >= 2749:
        return 1 # Low
    else:
        return 0 # High

X['Elevation'] = X['Elevation'].apply(converterElevation)

# Defining if Slope high or low
def converterSlope(column):
    if column >= 16.5:
        return 1 # Low
    else:
        return 0 # High

X['Slope'] = X['Slope'].apply(converterSlope)
# Normalizing numerical features so that each feature has mean 0 and variance 1

 

feature_scaler = StandardScaler()
X1 = feature_scaler.fit_transform(X)

# Analysis on subset1 - Personal Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# # Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X1)

# # Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =50,n_iter=1000)
x_tsne = tsne.fit_transform(X1)

elevation= list(X['Elevation'])
slope = list(X['Slope'])
hdth = list(X['Horizontal_Distance_To_Hydrology'])
vdth = list(X['Vertical_Distance_To_Hydrology'])
hdtr = list(X['Horizontal_Distance_To_Roadways'])
hdtf = list(X['Horizontal_Distance_To_Fire_Points'])
tree = list(X['Tree'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Elev: {a}; Slope: {b}; HDTH:{c}, VDTH:{d};HDTR: {e}; HDTF:{f}, Tree:{g}' for a,b,c,d,e,f,g in list(zip(elevation,slope,hdth,vdth,hdtr,hdtf,tree))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='ex2-50p-1000i-c2.html')
