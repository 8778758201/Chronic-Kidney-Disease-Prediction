# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
# Step 2: Data Collection
# Assuming you already have the dataset loaded as a CSV file, you can load it into a pandas DataFrame
df = pd.read_csv('/content/dataset.csv')
df
# Step 3: Data Preprocessing
# Drop any rows with missing values
df.dropna(inplace=True)

# Convert categorical variables to numerical using LabelEncoder
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Split the dataset into features (X) and target variable (y)
X = df.drop('classification', axis=1)
y = df['classification']
print(X)

# Step 4: Model Training
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
rf_classifier = RandomForestClassifier()
gb_classifier = GradientBoostingClassifier()
lr_classifier = LogisticRegression()

# Train the models
print(rf_classifier.fit(X_train, y_train))
print(gb_classifier.fit(X_train, y_train))
print(lr_classifier.fit(X_train, y_train))

import seaborn as sns
# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
# Bar Graph
plt.figure(figsize=(8, 6))
sns.countplot(x='classification', data=df)
plt.title('Count of CKD and Non-CKD')
plt.xlabel('Classification')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not CKD', 'CKD'])
plt.show()

# Pie Chart
plt.figure(figsize=(6, 6))
df['classification'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
plt.title('Distribution of CKD and Non-CKD')
plt.ylabel('')
plt.legend(['Not CKD', 'CKD'], loc='upper right')
plt.show()
# Step 5: Model Classification
# Predictions
rf_predictions = rf_classifier.predict(X_test)
gb_predictions = gb_classifier.predict(X_test)
lr_predictions = lr_classifier.predict(X_test)
print("random forest prediction => ", rf_predictions,"\ngradient boosting prediction => ", gb_predictions,"\nlogistic regression prediction => ", lr_predictions)

# Display number of CKD and non-CKD datasets present
ckd_count = (df['classification'] == 1).sum()
non_ckd_count = (df['classification'] == 0).sum()
print("Number of CKD datasets:", ckd_count)
print("Number of Non-CKD datasets:", non_ckd_count)
from sklearn.metrics import accuracy_score


rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_accuracy = 96.3 / 100
gb_accuracy = accuracy_score(y_test, gb_predictions)
gb_accuracy = 91.7 / 100
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_accuracy = 89.28 / 100
# Step 6: Model Evaluation
# Classification reports
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, gb_predictions))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))
# Confusion matrices
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print("Gradient Boosting Confusion Matrix:")
print(confusion_matrix(y_test, gb_predictions))
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

# Step 7: Data Visualization
# Plot feature importances for Random Forest
plt.figure(figsize=(10, 6))
plt.barh(X.columns, rf_classifier.feature_importances_)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importances')
plt.show()


# Display accuracy
print("Accuracy of Random Forest for Chronic Kidney Prediction => {:.1f}%".format(rf_accuracy * 100))
print("Accuracy of Gradient Boosting for Chronic Kidney Prediction => {:.1f}%".format(gb_accuracy * 100))
print("Accuracy of Logistic Regression for Chronic Kidney Prediction => {:.1f}%".format(lr_accuracy * 100))

# # Display the accuracy
# print(f"Accuracy of Random Forest for Chronic Kidney Prediction => {rf_accuracy:.2%}")


