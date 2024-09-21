import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Function to extract color histogram features from an image
def extract_features(image_name):
    image = cv2.imread(image_name)
    if image is None:
        print(f"Error: Image {image_name} not found.")
        return None
    # Resize the image for consistent feature extraction
    image = cv2.resize(image, (100, 100))
    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate the histogram
    histogram = cv2.calcHist([hsv_image], [0, 1], None, [8, 8], [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

# Create dataset
image_names = ["image1.jpeg", "image2.jpeg", "image3.jpeg", "image4.jpeg"]
labels = [0, 0, 1, 1]  # 0 for healthy, 1 for diseased

# Extract features
features = []
for image_name in image_names:
    feature = extract_features(image_name)
    if feature is not None:
        features.append(feature)
        print(f"Extracted features from {image_name}: {feature[:5]}...")  # Print first 5 feature values for debug

# Check if we have enough features to proceed
if len(features) < 2:
    print("Not enough data to train the classifier.")
else:
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train the SVM classifier
    clf = svm.SVC(kernel='linear')  # You can change the kernel if needed
    clf.fit(X_train, y_train)

    # Test the classifier
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Visualize the features
    plt.figure(figsize=(10, 5))
    for i in range(len(features)):
        plt.plot(features[i], label=f"{'Healthy' if labels[i] == 0 else 'Diseased'}: {image_names[i]}")
    plt.title('Feature Histograms')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.show()

    # Function to predict a new image
    def predict_leaf_disease(image_name):
        feature = extract_features(image_name)
        if feature is not None:
            prediction = clf.predict([feature])
            return "Healthy" if prediction[0] == 0 else "Diseased"
        return None

    # Example usage with a new image
    result = predict_leaf_disease("image4.jpeg")  # Replace with your actual image name
    print(f"The leaf is: {result}")




