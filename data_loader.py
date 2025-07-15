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

# Load datasets
df = pd.read_csv(r"D:\edu\Semester 8th\Data Mining\Blood_samples_dataset_balanced_2(f).csv")
test_data = pd.read_csv(r"D:\edu\Semester 8th\Data Mining\blood_samples_dataset_test.csv")
n_rows, n_columns = df.shape
print(" Number of objects: ",n_rows)
print(" Number of attributes: ",n_columns)
attributes = list(df.columns)
print(" Label set is: ", attributes[-1])
labels = set(df.iloc[:, -1])
print(" Labels are:\n", labels)
print(" Numeric attributes:\n", attributes[:-1])
df.describe()

df.info()

print('The number of people who might be healthy or have a particular disease :')
df['Disease'].value_counts()