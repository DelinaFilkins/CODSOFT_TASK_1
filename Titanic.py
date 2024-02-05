#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset (assuming it's a CSV file)
# Replace 'your_dataset.csv' with the actual file name
titanic_data = pd.read_csv('Titanic-Dataset.csv')

# Explore the dataset to understand its structure
# For example: titanic_data.info(), titanic_data.head(), etc.

# Drop irrelevant columns (e.g., PassengerId, Name, Ticket, Cabin)
titanic_data = titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values (you might need to customize this based on your dataset)
titanic_data = titanic_data.dropna()

# Convert categorical variables to numerical using one-hot encoding
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'])

# Define features (X) and target variable (y)
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Additional evaluation metrics
print(classification_report(y_test, y_pred))

# You can now use the trained model to make predictions on new data
# For example: model.predict(new_data)


# In[ ]:




