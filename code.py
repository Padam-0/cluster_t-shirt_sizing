import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns

df = pd.read_csv('data.csv', index_col=0)

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

shoulder_data = df[['ERECT SITTING HEIGHT', 'AGE IN MONTHS']]
shoulder_data['YEAR'] = shoulder_data['AGE IN MONTHS'].apply(lambda x: x//12)
x = shoulder_data.groupby('YEAR').mean().index[1:-2]
y = shoulder_data.groupby('YEAR').mean()['ERECT SITTING HEIGHT'][1:-2]

sns.set_style("white")
plt.rc('font', family='Raleway')

plt.scatter(x, y, c='black', s=20)

sns.despine()
plt.xlabel('Age (Years)')
plt.ylabel('Erect Sitting Height (mm)')
plt.xlim([0, 21])
plt.xticks([0, 5, 10, 15, 20])
plt.show()


shoulder_data = df[['ERECT SITTING HEIGHT', 'AGE IN MONTHS', 'SEX']]
shoulder_data['YEAR'] = shoulder_data['AGE IN MONTHS'].apply(lambda x: x//12)
y_1 = shoulder_data[shoulder_data['SEX'] == 1].groupby('YEAR').mean()['ERECT ' \
                        'SITTING HEIGHT'][1:-2]
y_2 = shoulder_data[shoulder_data['SEX'] == 2].groupby('YEAR').mean()['ERECT ' \
                        'SITTING HEIGHT'][:-1]
x = list(shoulder_data.groupby('YEAR').mean().index[1:-2])

sns.set_style("white")
plt.rc('font', family='Raleway')

plt.scatter(x, y_1, c='blue', s=20)
plt.scatter(x, y_2, c='red', s=20)

plt.legend(labels=["Sex = 1 (Male?)", "Sex = 2 (Female?)"],
               bbox_to_anchor=(0.05, .9, 0., 0.), loc=3, mode="expand",
               borderaxespad=0., markerscale=1)

sns.despine()
plt.xlabel('Age (Years)')
plt.ylabel('Erect Sitting Height (mm)')
plt.xlim([0, 21])
plt.xticks([0, 5, 10, 15, 20])
plt.show()


shoulder_data = df[['CHEST CIRCUMFERENCE', 'AGE IN MONTHS', 'SEX']]
shoulder_data['YEAR'] = shoulder_data['AGE IN MONTHS'].apply(lambda x: x//12)
y_1 = shoulder_data[shoulder_data['SEX'] == 1].groupby('YEAR').mean()['CHEST CIRCUMFERENCE'][1:-2]
y_2 = shoulder_data[shoulder_data['SEX'] == 2].groupby('YEAR').mean()['CHEST CIRCUMFERENCE'][:-1]
x = list(shoulder_data.groupby('YEAR').mean().index[1:-2])

sns.set_style("white")
plt.rc('font', family='Raleway')

plt.scatter(x, y_1, c='blue', s=20)
plt.scatter(x, y_2, c='red', s=20)

plt.legend(labels=["Sex = 1 (Male?)", "Sex = 2 (Female?)"],
               bbox_to_anchor=(0.05, .9, 0., 0.), loc=3, mode="expand",
               borderaxespad=0., markerscale=1)

sns.despine()
plt.xlabel('Age (Years)')
plt.ylabel('Chest Circumference (mm)')
plt.xlim([0, 21])
plt.xticks([0, 5, 10, 15, 20])
plt.show()

t_shirt_columns = ['CHEST CIRCUMFERENCE', 'WAIST CIRCUMFERENCE',
                   'SHOULDER-ELBOW LENGTH', 'ERECT SITTING HEIGHT',
                   'SHOULDER BREADTH']

young = df[df['AGE IN MONTHS'] < 84].loc[:,t_shirt_columns]
middle = df[(df['AGE IN MONTHS'] < 156) & (df['AGE IN MONTHS'] >= 83)].loc[:,t_shirt_columns]
old_1 = df[(df.SEX == 1) & (df['AGE IN MONTHS'] >= 156)].loc[:,t_shirt_columns]
old_2 = df[(df.SEX == 2) & (df['AGE IN MONTHS'] >= 156)].loc[:,t_shirt_columns]

drop_list = []
for i in range(len(middle.index)):
    if 0 in middle.ix[i,:].values:
        drop_list.append(i)

middle = middle.drop(middle.index[drop_list])

data = middle.as_matrix()

sns.set_style("white")
plt.rc('font', family='Raleway')
inertias = []
ks = range(1,11)
for k in ks:
    inertias.append(KMeans(n_clusters=k).fit(data).inertia_)
plt.plot(ks, inertias, '-o')
plt.xticks(ks)
plt.xlabel('Number of Clusters')
plt.ylabel('Total Inertia')
plt.show()


mergers = linkage(data, method='complete')
plt.figure(figsize=(20,10))
sns.set_style("white")
plt.rc('font', family='Raleway')
dendrogram(mergers, leaf_rotation=90, leaf_font_size=6)
plt.title("Middle Anthropometric T-shirt attributes Dendrogram")
plt.show()

kmeans = KMeans(n_clusters=4).fit(data)
samples = kmeans.predict(data)

names = middle.index

ns = pd.DataFrame({'names': names, 'cluster': samples})
clusters = ns.set_index(names).iloc[:,0]

print(clusters.value_counts())
print(kmeans.cluster_centers_)