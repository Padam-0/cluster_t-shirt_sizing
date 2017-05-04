import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import pylab

df = pd.read_csv('q5.csv', index_col=0)

remove_cols = []

for i in df.columns:
    if 3900 - df.loc[:,i].astype(bool).sum() > 2000:
        remove_cols.append(i)

df = df.drop(remove_cols, axis = 'columns')

demographic_attributes = ['AGE IN YEARS', 'LOCATION',
       'BIRTH DATE', 'MEASUREMENT DATE', 'MEASUREMENT SET TP',
       'MEASURER NUMBER', 'COMPUTER NUMBER', 'RACE', 'GRADE LEVEL',
       'HANDEDNESS', 'NUMBER OF BROTHERS', 'NUMBER OF SISTERS', 'TWIN',
       'BIRTH ORDER', 'MOTHERS OCCUPATION', 'FATHERS OCCUPATION',
       'MOTHERS EDUCATION', 'FATHERS EDUCATION', 'YEARS IN COMMUNITY',
       'ANTHROPOMETER NO', 'CALIPER NO', 'GIRTH NO']

df = df.drop(demographic_attributes, axis = 'columns')

print(df.loc[:,'AGE IN MONTHS'].describe())

young_1 = df[(df.SEX == 1) & (df['AGE IN MONTHS'] < 168)]
young_2 = df[(df.SEX == 2) & (df['AGE IN MONTHS'] >= 168)]
old_1 = df[(df.SEX == 1) & (df['AGE IN MONTHS'] < 168)]
old_2 = df[(df.SEX == 2) & (df['AGE IN MONTHS'] >= 168)]

t_shirt_columns = ['CHEST CIRCUMFERENCE', 'WAIST CIRCUMFERENCE',
                   'SHOULDER-ELBOW LENGTH', 'ERECT SITTING HEIGHT',
                   'SHOULDER BREADTH']

data = young_1.loc[:,t_shirt_columns].as_matrix()

mergers = linkage(data, method='complete')
dendrogram(mergers, labels = names, leaf_rotation=90, leaf_font_size=6)
plt.title("young_1 Anthropometric T-shirt attributes Dendrogram")
plt.show()

inertias = []
ks = range(1,11)
for k in ks:
    inertias.append(KMeans(n_clusters=k).fit(data).inertia_)
plt.plot(ks, inertias, '-o')
plt.xticks(ks)
plt.show()

kmeans = KMeans(n_clusters=5).fit(data)
samples = kmeans.predict(data)

x = data[:,0]
y = data[:,1]
plt.scatter(x, y, c=samples, cmap=pylab.cm.cool)
plt.title("KMeans")
centroids = kmeans.cluster_centers_
centroid_x = centroids[:,0]
centroid_y = centroids[:,1]
plt.scatter(centroid_x, centroid_y, marker='D', s=50, c='red')
plt.xlabel('CHEST CIRCUMFERENCE')
plt.ylabel('WAIST CIRCUMFERENCE')
plt.show()

samples = AffinityPropagation().fit_predict(data)
x = data[:,0]
y = data[:,1]
plt.scatter(x, y, c=samples, cmap=pylab.cm.cool)
plt.title("Affinity Propagation")
plt.xlabel('CHEST CIRCUMFERENCE')
plt.ylabel('WAIST CIRCUMFERENCE')
plt.show()

samples = MeanShift().fit_predict(data)
x = data[:,0]
y = data[:,1]
plt.scatter(x, y, c=samples, cmap=pylab.cm.cool)
plt.title("Mean Shift")
plt.xlabel('CHEST CIRCUMFERENCE')
plt.ylabel('WAIST CIRCUMFERENCE')
plt.show()

names = young_1.index

ns = pd.DataFrame({'names': names, 'cluster': samples})
clusters = ns.set_index(names).iloc[:,0]

print(clusters.value_counts())

print(kmeans.cluster_centers_)
