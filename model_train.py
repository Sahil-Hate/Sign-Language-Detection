import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib as joblib

# Define paths to the input and output directories
input_dir = 'C:\\Users\\Sahil\\Downloads\\test\\hog_features'
output_dir = 'C:\\Users\\Sahil\\Downloads\\test\\SVM classifier'

# Define the function to load features from disk
def load_features(features_dir):
    X = []
    y = []
    for class_dirname in os.listdir(features_dir):
        class_dir = os.path.join(features_dir, class_dirname)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith('.npy'):
                    features = np.load(os.path.join(class_dir, filename))
                    X.append(features)
                    y.append(class_dirname)
    return np.array(X), np.array(y)

# Load the features from the input directory
X, y = load_features(input_dir)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Test set accuracy: {accuracy:.2f}')

# Save the trained classifier to disk
clf_filename = os.path.join(output_dir, 'svm_classifier.joblib')
joblib.dump(clf, clf_filename)
