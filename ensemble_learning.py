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
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier


estimator = [] 
estimator.append(('LR',LogisticRegression(solver ='lbfgs',multi_class ='multinomial',max_iter = 200))) 
estimator.append(('GNB', GaussianNB())) 
estimator.append(('DTC', DecisionTreeClassifier(max_depth=7 , min_samples_split=3 , min_samples_leaf=3))) 
estimator.append(('ANN', MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42))) 
estimator.append(('SVM', SVC(probability=True))) 
estimator.append(('RF', RandomForestClassifier(n_estimators=100 , max_depth=8)))
estimator.append(('KNN', KNeighborsClassifier(n_neighbors=3)))  

hard_voting = VotingClassifier(estimators = estimator, voting ='hard') 
hard_voting.fit(X_train, y_train) 
y_pred_train_hard_voting = hard_voting.predict(X_train)  
y_pred_test_hard_voting = hard_voting.predict(X_test)  

print ('accuracy score train :', accuracy_score(y_train , y_pred_train_hard_voting))
print ('accuracy score test :', accuracy_score(y_test , y_pred_test_hard_voting))

print('precision score train :', precision_score(y_train, y_pred_train_hard_voting, average='weighted'))
print('precision score test :', precision_score(y_test, y_pred_test_hard_voting, average='weighted'))

print('recall score train :', recall_score(y_train, y_pred_train_hard_voting, average='weighted'))
print('recall score test :', recall_score(y_test, y_pred_test_hard_voting, average='weighted'))

print('f1 score train : ', f1_score(y_train, y_pred_train_hard_voting, average='weighted'))
print('f1 score test : ', f1_score(y_test, y_pred_test_hard_voting, average='weighted'))

#print('confusion matrix train :','\n',confusion_matrix(y_train , y_pred_train_hard_voting))
plt.figure(figsize=(8, 6))
disp_train = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train , y_pred_train_hard_voting), display_labels=['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc'])
disp_train.plot(cmap='Blues', ax=plt.gca())
plt.title("Confusion Matrix - Train Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.grid(False)
plt.show()

# Plot confusion matrix for test set

#print('confusion matrix test :','\n',confusion_matrix(y_test , y_pred_train_hard_voting))
plt.figure(figsize=(8, 6))
disp_test = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test , y_pred_test_hard_voting) , display_labels=['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc'])
disp_test.plot(cmap='Blues', ax=plt.gca())
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.grid(False)
plt.show()

soft_voting = VotingClassifier(estimators = estimator, voting ='soft') 
soft_voting.fit(X_train, y_train) 
y_pred_train_soft_voting = soft_voting.predict(X_train)  
y_pred_test_soft_voting = soft_voting.predict(X_test) 

print ('accuracy score train :', accuracy_score(y_train , y_pred_train_soft_voting))
print ('accuracy score test :', accuracy_score(y_test , y_pred_test_soft_voting))

print('precision score train :', precision_score(y_train, y_pred_train_soft_voting, average='weighted'))
print('precision score test :', precision_score(y_test, y_pred_test_soft_voting, average='weighted'))

print('recall score train :', recall_score(y_train, y_pred_train_soft_voting, average='weighted'))
print('recall score test :', recall_score(y_test, y_pred_test_soft_voting, average='weighted'))

print('f1 score train : ', f1_score(y_train, y_pred_train_soft_voting, average='weighted'))
print('f1 score test : ', f1_score(y_test, y_pred_test_soft_voting, average='weighted'))

#print('confusion matrix train :','\n',confusion_matrix(y_train , y_pred_train_hard_voting))
plt.figure(figsize=(8, 6))
disp_train = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train , y_pred_train_soft_voting), display_labels=['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc'])
disp_train.plot(cmap='Blues', ax=plt.gca())
plt.title("Confusion Matrix - Train Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.grid(False)
plt.show()

# Plot confusion matrix for test set

#print('confusion matrix test :','\n',confusion_matrix(y_test , y_pred_train_hard_voting))
plt.figure(figsize=(8, 6))
disp_test = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test , y_pred_test_soft_voting) , display_labels=['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc'])
disp_test.plot(cmap='Blues', ax=plt.gca())
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.grid(False)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


estimators = [
    ('svc', SVC(kernel='poly', probability=True)),
    ('dt', DecisionTreeClassifier(max_depth=7 , min_samples_split=3 , min_samples_leaf=3))
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stacking_clf.fit(X_train, y_train)


y_pred_train_stacking_clf = stacking_clf.predict(X_train)
y_pred_test_stacking_clf = stacking_clf.predict(X_test)

print ('accuracy score train :', accuracy_score(y_train , y_pred_train_stacking_clf))
print ('accuracy score test :', accuracy_score(y_test , y_pred_test_stacking_clf))

print('precision score train :', precision_score(y_train, y_pred_train_stacking_clf, average='weighted'))
print('precision score test :', precision_score(y_test, y_pred_test_stacking_clf, average='weighted'))

print('recall score train :', recall_score(y_train, y_pred_train_stacking_clf, average='weighted'))
print('recall score test :', recall_score(y_test, y_pred_test_stacking_clf, average='weighted'))

print('f1 score train : ', f1_score(y_train, y_pred_train_stacking_clf, average='weighted'))
print('f1 score test : ', f1_score(y_test, y_pred_test_stacking_clf, average='weighted'))

#print('confusion matrix train :','\n',confusion_matrix(y_train , y_pred_train_hard_voting))
plt.figure(figsize=(8, 6))
disp_train = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train , y_pred_train_stacking_clf), display_labels=['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc'])
disp_train.plot(cmap='Blues', ax=plt.gca())
plt.title("Confusion Matrix - Train Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.grid(False)
plt.show()

# Plot confusion matrix for test set

#print('confusion matrix test :','\n',confusion_matrix(y_test , y_pred_train_hard_voting))
plt.figure(figsize=(8, 6))
disp_test = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test , y_pred_test_stacking_clf) , display_labels=['Anemia', 'Diabetes', 'Healthy', 'Thalasse', 'Thromboc'])
disp_test.plot(cmap='Blues', ax=plt.gca())
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.grid(False)
plt.show()
