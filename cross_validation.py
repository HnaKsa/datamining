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
from gaussianNB import model_nb
from random_forest import rf
from dt_model import dt
from lr_model import model_LR
from kneighbors import knn
from svm import svm_model
from ann_model import model_ann
# Cross-validation and performance metrics for all models
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Example all_models dictionary, you should replace it with your actual models

all_models = {'dt_model': dt, 'knn_model': knn, 'lr_model': model_LR,
              'svm_model': svm_model, 'rf_model': rf, 'nb_model': model_nb , 'ann_model': model_ann}


for name, model in all_models.items():
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    print(f"{name} Cross-Validation Accuracy Scores:\n", cv_scores)
    print(f"{name} Mean Cross-Validation Accuracy:\n", np.mean(cv_scores))
    print("\n")

    # Cross-validation predictions
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=kfold)
    
    # Evaluate the model on cross-validated predictions (using training data)
    print(f"{name} Cross-Validation Accuracy: {accuracy_score(y_train, y_pred_cv)}")
    print(f"{name} Cross-Validation Precision: {precision_score(y_train, y_pred_cv, average='weighted')}")
    print(f"{name} Cross-Validation Recall: {recall_score(y_train, y_pred_cv, average='weighted')}")
    print(f"{name} Cross-Validation F1 Score: {f1_score(y_train, y_pred_cv, average='weighted')}")
    
    '''    # Generate a classification report
        print(f"Overall Classification Report for {name}:\n", classification_report(y_train, y_pred_cv, zero_division=0))'''

    # Confusion matrix
    conf_matrix = confusion_matrix(y_train, y_pred_cv)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc'], 
                yticklabels=['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

    print("\n\n")