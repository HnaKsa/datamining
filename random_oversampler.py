import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold , cross_val_predict
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from data_loader import df , test_data 
from preprocessing import X_train , y_train , X_test , y_test
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter
from sklearn.metrics import classification_report
ros = RandomOverSampler(random_state=101)
X_train_ros, y_train_ros= ros.fit_resample(X_train, y_train)
print(sorted(Counter(y_train_ros).items()))

ros_model = ros.fit(X_train_ros, y_train_ros)
y_pred_train_ros = ros_model.predict(X_train)
y_pred_test_ros = ros_model.predict(X_test)

print ('accuracy score train :', accuracy_score(y_train , y_pred_train_ros))
print ('accuracy score test :', accuracy_score(y_test , y_pred_test_ros))

print('precision score train :', precision_score(y_train, y_pred_train_ros, average='weighted'))
print('precision score test :', precision_score(y_test, y_pred_test_ros, average='weighted'))

print('recall score train :', recall_score(y_train, y_pred_train_ros, average='weighted'))
print('recall score test :', recall_score(y_test, y_pred_test_ros, average='weighted'))

print('f1 score train : ', f1_score(y_train, y_pred_train_ros, average='weighted'))
print('f1 score test : ', f1_score(y_test, y_pred_test_ros, average='weighted'))

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote= smote.fit_resample(X_train, y_train)

print(sorted(Counter(y_train_smote).items()))

smote_model = ros.fit(X_train_smote, y_train_smote)

y_pred_train_smote = smote_model.predict(X_train)
y_pred_test_smote = smote_model.predict(X_test)
# Check the model performance

print ('accuracy score train :', accuracy_score(y_train , y_pred_train_smote))
print ('accuracy score test :', accuracy_score(y_test , y_pred_test_smote))

print('precision score train :', precision_score(y_train, y_pred_train_smote, average='weighted'))
print('precision score test :', precision_score(y_test, y_pred_test_smote, average='weighted'))

print('recall score train :', recall_score(y_train, y_pred_train_smote, average='weighted'))
print('recall score test :', recall_score(y_test, y_pred_test_smote, average='weighted'))

print('f1 score train : ', f1_score(y_train, y_pred_train_smote, average='weighted'))
print('f1 score test : ', f1_score(y_test, y_pred_test_smote, average='weighted'))

