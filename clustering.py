from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def energy_clustering():
    sns.set_style('darkgrid')
    sns.color_palette=('hls', 5)
    matplotlib.rcParams['font.size'] = 9
    matplotlib.rcParams['figure.figsize'] = (6, 6)
    matplotlib.rcParams['figure.facecolor'] = '#00000000'

    my_data = pd.read_csv('Energy_2021_Data_Reported_in_2022_Excel_Prepared.csv')
    X = my_data[['Site_EUI_kBtu_per_sq_ft', 'Source_EUI_kBtu_per_sq_ft', 'Weather_Normalized_Site_EUI_kBtu_per_sq_ft', 'Weather_Normalized_Source_EUI_kBtu_per_sq_ft']]

    print("\n Features with highest correlations: ")
    print(X.corr(numeric_only=True))

    kmeans = KMeans(n_clusters = 8, random_state = 0, n_init = "auto").fit(X)
    preds = kmeans.predict(X)

    options = range(3,15)
    inertias = []

    for n_clusters in options:
        model = KMeans(n_clusters, random_state=0, n_init = "auto").fit(X)
        inertias.append(model.inertia_)
        
    plt.title("No. of clusters vs. Inertia")
    plt.xlabel('No. of clusters (K)')
    plt.ylabel('Inertia')
    plt.plot(options, inertias, '-o')
    matplotlib.pyplot.show()

    sns.scatterplot(data=my_data, x='Site_EUI_kBtu_per_sq_ft', y='Source_EUI_kBtu_per_sq_ft', hue=my_data['Energy_Rating'])
    centers_x, centers_y = kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1]
    plt.title('Distribution of Energy colored by Energy Rating')
    plt.plot(centers_x, centers_y, 'xr')
    matplotlib.pyplot.show()

    sns.scatterplot(data=X, y='Source_EUI_kBtu_per_sq_ft', x='Site_EUI_kBtu_per_sq_ft', hue=kmeans.labels_, palette=[ 'turquoise', 'g', 'cyan', 'steelblue', 'forestgreen', 'navy', 'grey', 'blue', 'lime', 'springgreen'])
    centers_x, centers_y = kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1]
    plt.title('Distribution of Energy colored by clusters')
    plt.plot(centers_x, centers_y, 'xr')
    matplotlib.pyplot.show()
