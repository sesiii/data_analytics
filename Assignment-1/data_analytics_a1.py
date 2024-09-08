# -*- coding: utf-8 -*-
"""Data_Analytics A1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ts3CHJc7ww-Pf-Q-Ge3gshgm5erwZO0m
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

df=pd.read_csv('content/Dataset - missing_values-SalaryData_Train.csv')

df.head()

"""# EDA"""

df.info()

df.shape

df.isnull().sum()

# prompt:  drop all the rows with nan value

df.dropna(inplace=True)

df['education'].unique()

# prompt: do label encoding of education column

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['education'] = le.fit_transform(df['education'])
df.head()

df['sex'].unique()

# prompt: do label encoding of sex column

df['sex'] = le.fit_transform(df['sex'])
df.head()

df['race'].unique()

# prompt: do label encoding of race column

df['race'] = le.fit_transform(df['race'])
df.head()

df.info()

# prompt: do label encoding of relationship column

df['relationship'].unique()

df['relationship'] = le.fit_transform(df['relationship'])
df.head()

# prompt: do frequency encoding of native column

# Calculate the frequency of each category in the 'native' column
freq_map = df['native'].value_counts(normalize=True).to_dict()

# Map the frequencies to the 'native' column
df['native_freq_encoded'] = df['native'].map(freq_map)

df.head()

# prompt: do frequency encoding of workclass column

# Calculate the frequency of each category in the 'workclass' column
freq_map = df['workclass'].value_counts(normalize=True).to_dict()

# Map the frequencies to the 'workclass' column
df['workclass_freq_encoded'] = df['workclass'].map(freq_map)

df.head()

# prompt: do frequency encoding of occupation column

# Calculate the frequency of each category in the 'occupation' column
freq_map = df['occupation'].value_counts(normalize=True).to_dict()

# Map the frequencies to the 'occupation' column
df['occupation_freq_encoded'] = df['occupation'].map(freq_map)

df.head()

import re

# Assuming 'df' is your DataFrame and 'Possibility' is the column name
df['Possibility'] = df['Possibility'].replace({'<=0.5': '0'}, regex=True)
df['Possibility'] = df['Possibility'].replace({'>0.5': '1'}, regex=True)

df.head()

# prompt: convert the datatpye of Possibility column to float

# Convert 'Possibility' column to float
df['Possibility'] = df['Possibility'].astype(float)

# prompt: do label encoding of maritalstatus column

df['maritalstatus'] = le.fit_transform(df['maritalstatus'])
df.head()

df.info()

# prompt: remove all the columns with dtype as object

# Select columns with data type other than object
df = df.select_dtypes(exclude=['object'])

df.head()

df.info()

# prompt: do univariate analsys of this df

# Univariate analysis for numerical features
df.describe()

# Histograms for numerical features
df.hist(figsize=(15, 10))
plt.show()

# Box plots for numerical features
df.boxplot(figsize=(15, 10))
plt.show()

# prompt: make a correlation matrix

# Correlation matrix
corr_matrix = df.corr()

# Plotting the correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# prompt: plot pie charts for the  columns education
# maritalstatus
# relationship
# race
# sex
# native_freq_encoded
# workclass_freq_encoded
# occupation_freq_encoded

# Pie charts for categorical features
plt.figure(figsize=(15, 10))
plt.subplot(3, 3, 1)
df['education'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Education')

plt.subplot(3, 3, 2)
df['maritalstatus'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Marital Status')

plt.subplot(3, 3, 3)
df['relationship'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Relationship')

plt.subplot(3, 3, 4)
df['race'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Race')

plt.subplot(3, 3, 5)
df['sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Sex')


plt.subplot(3, 3, 7)
df['workclass_freq_encoded'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Workclass Frequency Encoded')

plt.subplot(3, 3, 8)
df['occupation_freq_encoded'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Occupation Frequency Encoded')

plt.tight_layout()
plt.show()

# prompt: give density plots of maritalstatus
# relationship
#  race
#  sex
#  native_freq_encoded
#  workclass_freq_encoded
#  occupation_freq_encoded

# Density plots for selected features
plt.figure(figsize=(15, 10))

plt.subplot(3, 3, 1)
sns.kdeplot(df['maritalstatus'], shade=True)
plt.title('Marital Status')

plt.subplot(3, 3, 2)
sns.kdeplot(df['relationship'], shade=True)
plt.title('Relationship')

plt.subplot(3, 3, 3)
sns.kdeplot(df['race'], shade=True)
plt.title('Race')

plt.subplot(3, 3, 4)
sns.kdeplot(df['sex'], shade=True)
plt.title('Sex')

plt.subplot(3, 3, 5)
sns.kdeplot(df['native_freq_encoded'], shade=True)
plt.title('Native Frequency Encoded')

plt.subplot(3, 3, 6)
sns.kdeplot(df['workclass_freq_encoded'], shade=True)
plt.title('Workclass Frequency Encoded')

plt.subplot(3, 3, 7)
sns.kdeplot(df['occupation_freq_encoded'], shade=True)
plt.title('Occupation Frequency Encoded')

plt.tight_layout()
plt.show()

# prompt:  Perform an Exploratory Data Analysis (EDA) on the dataset. EDA may include
#  frequency distribution, and multivariate correlation analysis, as well as basic
#  data visualization

# Frequency distribution for categorical features
for col in df.select_dtypes(include=['int64']):
  print(f"\nFrequency distribution for {col}:")
  print(df[col].value_counts())


# Pairplot for visualizing relationships between numerical features
sns.pairplot(df)
plt.show()

# Box plots for numerical features against a categorical feature
for col in df.select_dtypes(include=['int64']):
  plt.figure()
  sns.boxplot(x='Possibility', y=col, data=df)
  plt.title(f'Boxplot of {col} vs Possibility')
  plt.show()

df_copy=df.copy()



"""# Naive Bayes model (Part-B)"""

# prompt:  Implement Naive Bayes model on the given dataset without relying on any machine
#  learning libraries (e.g., sklearn). Your task is to code the Naive Bayes algorithm from
#  scratch to classify individuals as either criminal(Possibility=1) or innocent(Possibility=0). You may use basic packages
#  such as numpy, pandas, and math.

import pandas as pd
import numpy as np
import math

def calculate_prior_probability(df, target_variable):
  """
  Calculates the prior probability of each class in the target variable.

  Args:
    df: Pandas DataFrame.
    target_variable: Name of the target variable column.

  Returns:
    A dictionary containing the prior probabilities for each class.
  """
  class_counts = df[target_variable].value_counts()
  total_samples = len(df)
  prior_probabilities = {cls: count / total_samples for cls, count in class_counts.items()}
  return prior_probabilities


def calculate_likelihood(df, feature, feature_value, target_variable, target_class):
  """
  Calculates the likelihood of a feature value given a target class.

  Args:
    df: Pandas DataFrame.
    feature: Name of the feature column.
    feature_value: Value of the feature.
    target_variable: Name of the target variable column.
    target_class: Target class value.

  Returns:
    The likelihood of the feature value given the target class.
  """
  filtered_df = df[(df[target_variable] == target_class) & (df[feature] == feature_value)]
  count = len(filtered_df)
  total_count = len(df[df[target_variable] == target_class])
  likelihood = count / total_count
  return likelihood


def naive_bayes_predict(df, features, target_variable, new_data_point):
  """
  Predicts the target class for a new data point using Naive Bayes.

  Args:
    df: Pandas DataFrame.
    features: List of feature column names.
    target_variable: Name of the target variable column.
    new_data_point: A dictionary containing the feature values for the new data point.

  Returns:
    The predicted class for the new data point.
  """
  prior_probabilities = calculate_prior_probability(df, target_variable)
  classes = df[target_variable].unique()
  predictions = {}

  for cls in classes:
    posterior_probability = prior_probabilities[cls]
    for feature in features:
      feature_value = new_data_point[feature]
      likelihood = calculate_likelihood(df, feature, feature_value, target_variable, cls)
      posterior_probability *= likelihood
    predictions[cls] = posterior_probability

  predicted_class = max(predictions, key=predictions.get)
  return predicted_class


# Select features and target variable
features = df.columns[:-1].tolist()  # Exclude the last column ('Possibility')
target_variable = 'Possibility'
if target_variable in features:
    features.remove(target_variable)

# Example usage:
new_data_point = {
    'age': 30,
    'workclass_freq_encoded': 0.5,
    'education': 1,
    'educationno':5.0,
    'maritalstatus': 1,
    'occupation_freq_encoded': 0.3,
    'relationship': 1,
    'race': 1,
    'sex': 1,
    'capitalgain': 0,
    'capitalloss': 0,
    'hoursperweek': 40,
    'native_freq_encoded': 0.8
}

predicted_class = naive_bayes_predict(df, features, target_variable, new_data_point)
print(f"Predicted class for the new data point: {predicted_class}")

# prompt: make a train and test set and check the accuracy of the above naive bayes model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X = df.drop('Possibility', axis=1)
y = df['Possibility']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a new DataFrame for training
df_train = pd.concat([X_train, y_train], axis=1)

# Predict on the test set
y_pred = []
for index, row in X_test.iterrows():
  new_data_point = row.to_dict()
  predicted_class = naive_bayes_predict(df_train, features, target_variable, new_data_point)
  y_pred.append(predicted_class)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Naive Bayes model: {accuracy}")

"""# Part-C"""

# prompt:  Now, implement Naive Bayes, SVM, Decision Tree, and KNN using the sklearn module
#  to perform the classification task. Compare the performance of the sklearn Naive Bayes
#  implementation with the custom Naive Bayes implementation

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Sklearn Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print(f"Accuracy of Sklearn Naive Bayes model: {accuracy_gnb}")

# SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy of SVM model: {accuracy_svm}")

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy of Decision Tree model: {accuracy_dt}")

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy of KNN model: {accuracy_knn}")

# Compare with custom Naive Bayes
print(f"Accuracy of Custom Naive Bayes model: {accuracy}")

"""# Part-D"""

import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

# Custom function to perform ensemble voting
def ensemble_predict(X, classifiers):
    predictions = []

    for classifier in classifiers:
        y_pred = classifier.predict(X)
        predictions.append(y_pred)

    # Transpose the predictions to align each instance's predictions from all classifiers
    predictions = np.array(predictions).T

    # Apply majority voting
    final_predictions = []
    for instance_preds in predictions:
        majority_vote = Counter(instance_preds).most_common(1)[0][0]
        final_predictions.append(majority_vote)

    return np.array(final_predictions)

# Wrapping the custom Naive Bayes model to fit into the ensemble
class CustomNaiveBayesWrapper:
    def predict(self, X):
        y_pred = []
        for _, row in X.iterrows():
            new_data_point = row.to_dict()
            predicted_class = naive_bayes_predict(df_train, features, target_variable, new_data_point)
            y_pred.append(predicted_class)
        return np.array(y_pred)

custom_nb = CustomNaiveBayesWrapper()

# Train sklearn models
svm = SVC()
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()

svm.fit(X_train, y_train)
dt.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Create a list of all classifiers
classifiers = [svm, dt, knn, custom_nb]

# Use the ensemble model to predict on the test set
y_pred_ensemble = ensemble_predict(X_test, classifiers)

# Evaluate the ensemble model
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
f1_score_ensemble = f1_score(y_test, y_pred_ensemble)

print(f"Accuracy of Ensemble model: {accuracy_ensemble}")
print(f"F1 Score of Ensemble model: {f1_score_ensemble}")

# Collect all model accuracies and F1 scores for comparison
accuracy_custom = accuracy  # Accuracy of custom Naive Bayes from Part B
f1_score_custom = f1_score(y_test, y_pred)  # F1 score for custom Naive Bayes

accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
f1_score_gnb = f1_score(y_test, y_pred_gnb)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_score_svm = f1_score(y_test, y_pred_svm)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_score_dt = f1_score(y_test, y_pred_dt)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_score_knn = f1_score(y_test, y_pred_knn)

# Model names and their corresponding accuracies and F1 scores
model_names = ['Custom Naive Bayes', 'Sklearn Naive Bayes', 'SVM', 'Decision Tree', 'KNN', 'Ensemble']
accuracies = [accuracy_custom, accuracy_gnb, accuracy_svm, accuracy_dt, accuracy_knn, accuracy_ensemble]
f1_scores = [f1_score_custom, f1_score_gnb, f1_score_svm, f1_score_dt, f1_score_knn, f1_score_ensemble]

# Plotting the comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(model_names, accuracies, color='skyblue', label='Accuracy')
plt.barh(model_names, f1_scores, color='orange', alpha=0.5, label='F1 Score')
plt.xlabel('Scores')
plt.ylabel('Models')
plt.title('Comparison of Models')
plt.legend()
plt.show()

