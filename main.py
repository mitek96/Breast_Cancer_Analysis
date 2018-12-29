import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics

# Przygotowanie danych
data = pd.read_csv("data.csv", header=0)
data.drop("Unnamed: 32", axis=1, inplace=True)
data.drop("id", axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
sns.countplot(data['diagnosis'], label="Count")
plt.savefig('count.png')


features_mean = list(data.columns[1:11])
features_se = list(data.columns[11:20])
features_worst = list(data.columns[21:31])

corr = data[features_mean].corr()
plt.figure(figsize=(14, 14))
sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2f', annot_kws={'size': 15},
            xticklabels=features_mean, yticklabels=features_mean,
            cmap='coolwarm')
plt.savefig('correlation.png')

prediction_var = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']

train, test = train_test_split(data, test_size=0.3)
train_X = train[prediction_var]
train_y = train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis

# Random Forest Classifier
model_RFC = RandomForestClassifier(n_estimators=100)
model_RFC.fit(train_X, train_y)
prediction_RFC = model_RFC.predict(test_X)
print("RandomForestClassifier accuracy: {}".format(metrics.accuracy_score(prediction_RFC, test_y)))


# Support Vector Classification
model_SVC = svm.SVC(gamma='auto')
model_SVC.fit(train_X, train_y)
prediction_SVC = model_SVC.predict(test_X)
print("SupportVectorClassification accuracy: {}".format(metrics.accuracy_score(prediction_SVC, test_y)))


color_function = {0: "blue", 1: "red"}  # Czerwony to złośliwy, niebieski Here Red color will be 1 which means M and blue foo 0 means B
colors = data["diagnosis"].map(lambda x: color_function.get(x))  # mapping the color fuction with diagnosis column
pd.plotting.scatter_matrix(data[features_mean], c=colors, alpha=0.5, figsize=(15, 15))  # plotting scatter plot matrix
plt.savefig('matrix.png')

# Wprowadzenie nowych danych pacjenta
tex_mean = input("texture_mean=")
per_mean = input("perimeter_mean=")
smo_mean = input("smoothness_mean=")
com_mean = input("compactness_mean=")
sym_mean = input("symmetry_mean")

single_data = test[prediction_var]
single_data = single_data[0:0]
single_data = single_data.append(
    pandas.Series([tex_mean, per_mean, smo_mean, com_mean, sym_mean], index=single_data.columns), ignore_index=True)

prediction_single = model.predict(single_data)
prediction_single = 'M' if 1 else 'B'
print(prediction_single)
