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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)

model_ann.fit(X_train, y_train)

y_pred_test_ann = model_ann.predict(X_test)
y_pred_train_ann = model_ann.predict(X_train)

# Plot confusion matrix for train set
plt.figure(figsize=(8, 6))
disp_train = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train , y_pred_train_ann), display_labels=['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc'])
disp_train.plot(cmap='Blues', ax=plt.gca())
plt.title("Confusion Matrix - Train Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.grid(False)
plt.show()

# Plot confusion matrix for test set
plt.figure(figsize=(8, 6))
disp_test = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test , y_pred_test_ann) , display_labels=['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc'])
disp_test.plot(cmap='Blues', ax=plt.gca())
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.grid(False)
plt.show()

print('accuracy score train :', accuracy_score(y_train, y_pred_train_ann))
print('accuracy score test :', accuracy_score(y_test, y_pred_test_ann))

print('precision score train :', precision_score(y_train, y_pred_train_ann, average='weighted'))
print('precision score test :', precision_score(y_test, y_pred_test_ann, average='weighted'))

print('recall score train :', recall_score(y_train, y_pred_train_ann, average='weighted'))
print('recall score test :', recall_score(y_test, y_pred_test_ann, average='weighted'))

print('f1 score train : ', f1_score(y_train, y_pred_train_ann, average='weighted'))
print('f1 score test : ', f1_score(y_test, y_pred_test_ann, average='weighted'))