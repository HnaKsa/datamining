# 🧠 Modular Data Mining and Machine Learning Project

This project is a modular and extensible implementation of various supervised machine learning algorithms applied to a classification problem. Each model and core functionality is encapsulated in its own Python module for reusability, scalability, and clarity.

---

## 📁 Project Structure

The project is organized into standalone Python files, each responsible for a specific part of the machine learning pipeline:

| File/Module                | Description |
|---------------------------|-------------|
| `data_loader.py`          | Loads and formats the dataset. |
| `preprocessing.py`        | Handles encoding, scaling, and train/test splitting. |
| `gaussianNB.py`           | Gaussian Naive Bayes model implementation. |
| `kneighbors.py`           | K-Nearest Neighbors (KNN) classifier. |
| `dt_model.py`             | Decision Tree classifier. |
| `random_forest.py`        | Random Forest classifier. |
| `svm.py`                  | Support Vector Machine (SVM) implementation. |
| `lr_model.py`             | Logistic Regression model. |
| `l1&2_regularization.py`  | Demonstrates L1 and L2 regularization techniques. |
| `random_oversampler.py`   | Handles imbalanced data using oversampling techniques (e.g., SMOTE). |
| `ensemble_learning.py`    | Implements ensemble methods like voting classifiers. |
| `cross_validation.py`     | Performs cross-validation to evaluate model performance. |
| `ann_model.py`            | Builds an Artificial Neural Network (ANN) model. |
| `visualizations/`         | Contains plots for EDA and model performance visualization. |
| `README.md`               | Project documentation (this file). |

---

## 🔍 Objective

The main goal is to compare the performance of various machine learning algorithms on a common classification task using a consistent preprocessing pipeline. The models are evaluated using metrics such as accuracy, confusion matrix, precision, recall, and F1-score.

---

## ⚙️ Key Features

- ✅ Modular design for each algorithm and process
- 🔄 Data preprocessing and feature engineering
- 🧪 Oversampling to address class imbalance
- ⚖️ Regularization techniques (L1 & L2)
- 🔁 Cross-validation for reliable evaluation
- 🤝 Ensemble learning methods for performance boosting
- 📈 Visualizations for interpretability
- 🔧 Easy integration or expansion with new models or datasets

---

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/HnaKsa/datamining.git  
cd datamining  
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Run a Module
Each script can be executed independently. Example:

```bash
python gaussianNB.py
python ensemble_learning.py
```
## 📊 Models Implemented
Gaussian Naive Bayes

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

Support Vector Machine (SVM)

Logistic Regression (with L1 & L2 Regularization)

Artificial Neural Network (ANN)

Voting Classifier (Ensemble)

## 📈 Evaluation Metrics
Accuracy

Confusion Matrix

Precision

Recall

F1-score

Cross-validation scores

## 📂 Dataset
The dataset is expected to be loaded via the data_loader.py module. Modify the file path or loading function to suit your specific dataset format (e.g., CSV, Excel).

## 📁 Visualizations
All model evaluation visuals are stored in the visualizations/ folder, including:

Confusion matrices

Accuracy comparison plots

ROC curves  

ANN training loss/accuracy graphs

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repo, open issues, or submit pull requests to improve features, models, or documentation.

## 📬 Contact
For questions or collaboration opportunities, please contact:  
Hanna Kasaei  
📧 [hannahkasaei@gmail.com.com]  
🔗 [my linkedin profile](https://www.linkedin.com/in/hnaksa?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BuVFWMgChQj6SZ46K79NYGg%3D%3D)  
🐙[my github](https://github.com/HnaKsa)  

## 📝 License
This project is licensed under the MIT License.
