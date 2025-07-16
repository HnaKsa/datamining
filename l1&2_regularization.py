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
from sklearn.linear_model import LogisticRegression

# Define the model with L1 regularization
model_lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=200)

# Fit the model
model_lr_l1.fit(X_train, y_train)

# Predictions
y_pred_train_lr_l1 = model_lr_l1.predict(X_train)
y_pred_test_lr_l1 = model_lr_l1.predict(X_test)

# Define the model with L2 regularization
model_lr_l2 = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=200)

# Fit the model
model_lr_l2.fit(X_train, y_train)

# Predictions
y_pred_train_lr_l2 = model_lr_l2.predict(X_train)
y_pred_test_lr_l2 = model_lr_l2.predict(X_test)

from sklearn.svm import LinearSVC

# Define the model with L1 regularization
model_svm_l1 = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=1.0, max_iter=200)

# Fit the model
model_svm_l1.fit(X_train, y_train)

# Predictions
y_pred_train_svm_l1 = model_svm_l1.predict(X_train)
y_pred_test_svm_l1 = model_svm_l1.predict(X_test)
# Define the model with L2 regularization (default)
model_svm_l2 = LinearSVC(penalty='l2', loss='squared_hinge', C=1.0, max_iter=200)

# Fit the model
model_svm_l2.fit(X_train, y_train)

# Predictions
y_pred_train_svm_l2 = model_svm_l2.predict(X_train)
y_pred_test_svm_l2 = model_svm_l2.predict(X_test)

# Example for Logistic Regression with L2 regularization
print('Logistic Regression with L2 Regularization:')
print('Train Accuracy:', accuracy_score(y_train, y_pred_train_lr_l2))
print('Test Accuracy:', accuracy_score(y_test, y_pred_test_lr_l2))
print('Train Precision:', precision_score(y_train, y_pred_train_lr_l2, average='weighted'))
print('Test Precision:', precision_score(y_test, y_pred_test_lr_l2, average='weighted'))
print('Train Recall:', recall_score(y_train, y_pred_train_lr_l2, average='weighted'))
print('Test Recall:', recall_score(y_test, y_pred_test_lr_l2, average='weighted'))
print('Train F1-score:', f1_score(y_train, y_pred_train_lr_l2, average='weighted'))
print('Test F1-score:', f1_score(y_test, y_pred_test_lr_l2, average='weighted'))

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the model with L1 regularization
model_svm_l1 = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, max_iter=200)

# Define the parameter grid for GridSearchCV
param_grid_l1 = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100]  # Example values for C to be tested
}

# Perform GridSearchCV with 5-fold cross-validation
grid_search_l1 = GridSearchCV(model_svm_l1, param_grid_l1, cv=5, scoring='accuracy', verbose=1)
grid_search_l1.fit(X_train, y_train)

# Get the best model and evaluate on test data
best_model_l1 = grid_search_l1.best_estimator_

# Predictions
y_pred_train_svm_l1 = best_model_l1.predict(X_train)
y_pred_test_svm_l1 = best_model_l1.predict(X_test)

# Evaluate the model
print('Linear SVM with L1 Regularization:')
print('Best Parameters:', grid_search_l1.best_params_)
print('Train Accuracy:', accuracy_score(y_train, y_pred_train_svm_l1))
print('Test Accuracy:', accuracy_score(y_test, y_pred_test_svm_l1))
print('Train Precision:', precision_score(y_train, y_pred_train_svm_l1, average='weighted'))
print('Test Precision:', precision_score(y_test, y_pred_test_svm_l1, average='weighted'))
print('Train Recall:', recall_score(y_train, y_pred_train_svm_l1, average='weighted'))
print('Test Recall:', recall_score(y_test, y_pred_test_svm_l1, average='weighted'))
print('Train F1-score:', f1_score(y_train, y_pred_train_svm_l1, average='weighted'))
print('Test F1-score:', f1_score(y_test, y_pred_test_svm_l1, average='weighted'))
