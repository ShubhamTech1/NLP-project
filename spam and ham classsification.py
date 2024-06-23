


import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv(r"D:\NLP, CV\NLP\spam.csv", encoding='latin1')

df.isnull().sum() 
columns_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
df.drop(columns=columns_to_drop, inplace=True)
df.isnull().sum() 


# Split the data into features and target
X = df['v2'] 
y = df['v1']


# convert categorical features into numrical 

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Encode the target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)



# Train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train_encoded)









from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, accuracy_score

# TRAINING ACCURACY:-
y_pred = model.predict(X_train_vectorized)  
training_accuracy = accuracy_score(y_train_encoded, y_pred)
training_accuracy


# TESTING ACCURACY:-
# Make predictions on the test set
y_test_pred = model.predict(X_test_vectorized)
testing_accuracy = accuracy_score(y_test_encoded, y_test_pred)
testing_accuracy

#---------------------------------------------------------------------------------------------------------

# check dataset is balanced or not.
from collections import Counter
# Assuming you have your data (X) and target variable (y)

# Get class labels and their counts
class_counts = Counter(y)

# Calculate total number of data points
total_data_points = len(y)

# Calculate percentage of each class
class_percentages = {class_label: (count / total_data_points) * 100 for class_label, count in class_counts.items()}

# Print class distribution information
print("Class Distribution:")
for class_label, percentage in class_percentages.items():
    print(f"{class_label}: {percentage:.2f}%")



#---------------------------------------------------------------------------------------------------------


# in case of spam classification problem we focus on only precision score.

# Calculate evaluation metrics
precision = precision_score(y_test_encoded, y_test_pred)  
print(f'Precision: {precision:.2f}')

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_encoded, y_test_pred))





import pandas as pd
# Assuming y_test_encoded is a pandas Series
y_test_encoded = pd.Series(y_test_encoded)

# Count occurrences of each class
class_counts = y_test_encoded.value_counts()

# Print the counts
print("Count of 0s:", class_counts.get(0)) 
print("Count of 1s:", class_counts.get(1))



# we build our best model bcoz our FP = 0
# if FP and FN both are equally important :- limitation(dataset is Balance,then check accuracy & F1_SCORE)








'''




recall = recall_score(y_test_encoded, y_test_pred)
print(f'Recall: {recall:.2f}')

f1 = f1_score(y_test_encoded, y_test_pred)
print(f'F1 Score: {f1:.2f}')

y_pred_prob = model.predict_proba(X_test_vectorized)[:, 1]  # Probability estimates for the positive class
roc_auc = roc_auc_score(y_test_encoded, y_pred_prob)
print(f'ROC AUC Score: {roc_auc:.2f}')

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_test_pred))

'''




#--------------------------------------------------------------------------------------------------------------------------------------------

# convert imbalanced dataset into balanced :-
# OVERSAMPLING(SMOTE) :- INCREASED THE DATA IF MINORITY CLASS WHICH WILL BE EQUAL TO THE MAJORITY CLASS.


from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_vectorized, y_train_encoded)



'''


import pandas as pd
# Assuming y_test_encoded is a pandas Series
y_train_smote = pd.Series(y_train_smote)

# Count occurrences of each class
class_counts = y_train_smote.value_counts()

# Print the counts
print("Count of 0s:", class_counts.get(0)) 
print("Count of 1s:", class_counts.get(1))

'''


'''

# check dataset is balanced or not.
from collections import Counter
# Assuming you have your data (X) and target variable (y)

# Get class labels and their counts
class_counts = Counter(y_train_smote)

# Calculate total number of data points
total_data_points = len(y_train_smote)

# Calculate percentage of each class
class_percentages = {class_label: (count / total_data_points) * 100 for class_label, count in class_counts.items()}

# Print class distribution information
print("Class Distribution:")
for class_label, percentage in class_percentages.items():
    print(f"{class_label}: {percentage:.2f}%")

# 50%-50%
'''









# NOW TRAIN OUR MODEL AFTER  OVERSAMPLING(SMOTE)

model = MultinomialNB()
model.fit(X_train_smote, y_train_smote)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, accuracy_score

# TRAINING ACCURACY:-
y_pred = model.predict(X_train_smote)  
training_accuracy = accuracy_score(y_train_smote, y_pred)
training_accuracy


# TESTING ACCURACY:-
# Make predictions on the test set
y_test_pred = model.predict(X_test_vectorized) 
testing_accuracy = accuracy_score(y_test_encoded, y_test_pred)
testing_accuracy


# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_encoded, y_test_pred))





# check dataset is balanced or not.
from collections import Counter
# Assuming you have your data (X) and target variable (y)

# Get class labels and their counts
class_counts = Counter(y_train_smote)

# Calculate total number of data points
total_data_points = len(y_train_smote)

# Calculate percentage of each class
class_percentages = {class_label: (count / total_data_points) * 100 for class_label, count in class_counts.items()}

# Print class distribution information
print("Class Distribution:")
for class_label, percentage in class_percentages.items():
    print(f"{class_label}: {percentage:.2f}%")









