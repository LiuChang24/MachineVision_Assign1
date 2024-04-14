import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from classifier import MahalanobisClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

# Read Dataset
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv', header=None)
df.dropna(inplace=True)  # Drop missing values.
df.head()

# split the dataset
X = df.iloc[:, 0:7].values  # Sample Data
y = df.iloc[:, 7].values  # Label Data
unique_y = np.unique(y)  # To return the correct labels when predicting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=100)
# split the dataset in 70:30 ratio as Train and Test

# Normalization
sc = StandardScaler()  # instantiate
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)  # Normalize the dataset

# PCA
pca = PCA(n_components=2) # 压缩到二维特征
X_train_pca = pca.fit_transform(X_train_std) # 对训练数据进行处理
X_test_pca = pca.transform(X_test_std) # 对测试集数据进行处理

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    # 按照样本的真实值进行展示
    plt.scatter(X_train_pca[y_train == l, 0],
                X_train_pca[y_train == l, 1],
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# "Training"
clf = MahalanobisClassifier(X_train_pca, y_train)

# Predicting
pred_probs = clf.predict_probability(X_test_pca)
pred_class = clf.predict_class(X_test_pca,unique_y)

pred_actuals = pd.DataFrame([(pred, act) for pred, act in zip(pred_class, y_test)], columns=['pred', 'true'])
print(pred_actuals[:5])

truth = pred_actuals.loc[:, 'true']
pred = pred_actuals.loc[:, 'pred']
scores = np.array(pred_probs)[:, 1]
print('\nAccuracy Score: ', accuracy_score(truth, pred))
print('\nClassification Report: \n', classification_report(truth, pred))

