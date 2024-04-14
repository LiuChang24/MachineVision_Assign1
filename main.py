import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from classifier import MahalanobisClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

# 读取数据
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv', header=None)
df.dropna(inplace=True)  # Drop missing values.
df.head()

# 数据集划分
X = df.iloc[:, 0:7].values  # Sample Data
y = df.iloc[:, 7].values  # Label Data
unique_y = np.unique(y)  # To return the correct labels when predicting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=100)
# split the dataset in 70:30 ratio as Train and Test

# Normalization
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.fit_transform(X_test_std, y_test)
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

plot_decision_regions(X_test_lda, y_test, lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# "Training"
clf = MahalanobisClassifier(X_train_lda, y_train)

# Predicting
pred_probs = clf.predict_probability(X_test_lda)
pred_class = clf.predict_class(X_test_lda,unique_y)

pred_actuals = pd.DataFrame([(pred, act) for pred, act in zip(pred_class, y_test)], columns=['pred', 'true'])
print(pred_actuals[:5])

truth = pred_actuals.loc[:, 'true']
pred = pred_actuals.loc[:, 'pred']
scores = np.array(pred_probs)[:, 1]
print('\nAccuracy Score: ', accuracy_score(truth, pred))
print('\nClassification Report: \n', classification_report(truth, pred))