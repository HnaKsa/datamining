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
from data_loader import df , test_data 

# Before encoding, we take a look at the specific values of the dataset columns
def unique_values(dataframe):
    unique_values_dict = {}
    for column in dataframe.columns:
        unique_values_dict[column] = set(dataframe[column])
    return unique_values_dict

print(unique_values(pd.DataFrame(df)))

# Handle missing values
row_nan_count = df.isnull().sum(axis=1)
print("Missing values per row:\n", row_nan_count)

# Encode labels
label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])

# Scale features
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

# Prepare training and test data
test_data = test_data.dropna()
df = df.dropna()
test_data['Disease'] = label_encoder.transform(test_data['Disease'])
test_data_scaled = pd.DataFrame(scaler.transform(test_data), columns = test_data.columns)

X_train = df.drop('Disease', axis = 1)
y_train = df['Disease']
X_test = test_data.drop('Disease', axis = 1)
y_test = test_data['Disease']