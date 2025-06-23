
#Task 1: Classical ML with Scikit-learn

# Iris Species Classification with Decision Tree
#
# This script loads the Iris dataset, preprocesses it (handles missing values and encodes labels),
# trains a Decision Tree Classifier, and evaluates it using accuracy, precision, and recall.

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 1. Load the Iris Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# For demonstration, let's assume there could be missing values. Introduce some randomly:
np.random.seed(42)
missing_mask = np.random.rand(*X.shape) < 0.05
X[missing_mask] = np.nan

# 2. Preprocessing

# 2.1 Handle missing values (using mean imputation)
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 2.2 Encode labels (not needed for sklearn's decision tree, but for generality)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Train Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 5. Predict on Test Data
y_pred = clf.predict(X_test)

# 6. Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # macro: equal weight per class
recall = recall_score(y_test, y_pred, average='macro')

print("Decision Tree Classifier on Iris Dataset")
print("Accuracy : {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall   : {:.4f}".format(recall))

# If you want to see a classification report for all classes:
from sklearn.metrics import classification_report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))



#Part 2: Practical Implementation

# MNIST Handwritten Digit Classification with CNN
#
# This script trains a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.
# It aims to achieve >95% test accuracy, and visualizes predictions for 5 test images.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Preprocess data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
# Add channel dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# One-hot encode labels
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# 3. Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
history = model.fit(
    x_train, y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

# 5. Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# 6. Visualize predictions on 5 sample images
sample_idx = np.random.choice(len(x_test), 5, replace=False)
sample_images = x_test[sample_idx]
sample_labels = y_test[sample_idx]
preds = model.predict(sample_images)
pred_digits = np.argmax(preds, axis=1)

plt.figure(figsize=(10,2))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(sample_images[i].reshape(28,28), cmap='gray')
    plt.title(f"Label: {sample_labels[i]}\nPred: {pred_digits[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# --- End of Script ---

# Notes:
# - This script uses TensorFlow/Keras.
# - It achieves >95% accuracy in a few epochs on MNIST.
# - To run: install tensorflow and matplotlib if needed.




#Task 3: NLP with spaCy

# NLP with spaCy: Named Entity Recognition and Sentiment Analysis on Amazon Reviews
#
# This script extracts product names and brands using spaCy's NER,
# and performs rule-based sentiment analysis (positive/negative)
# on sample Amazon product reviews.

import spacy

# 1. Load English spaCy model
nlp = spacy.load("en_core_web_sm")

# 2. Sample Amazon product reviews
reviews = [
    "I love my new Samsung Galaxy phone! The camera quality is amazing.",
    "The Nike running shoes were not comfortable and felt cheap.",
    "Apple's MacBook Pro is excellent for work, but the battery life could be better.",
    "I am disappointed with the Sony headphones. The sound quality is poor.",
    "Logitech mouse works perfectly and is very affordable."
]

# 3. Define simple positive/negative word lists for rule-based sentiment
positive_words = {"love", "amazing", "excellent", "perfectly", "affordable"}
negative_words = {"not", "disappointed", "poor", "cheap"}

def simple_sentiment(text):
    text_lower = text.lower()
    if any(word in text_lower for word in positive_words):
        return "Positive"
    if any(word in text_lower for word in negative_words):
        return "Negative"
    return "Neutral"

# 4. Process each review
for i, review in enumerate(reviews, 1):
    doc = nlp(review)
    # Extract product names and brands (entities labeled as ORG or PRODUCT)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in {"PRODUCT", "ORG"}]
    sentiment = simple_sentiment(review)
    print(f"Review {i}: {review}")
    print("  Extracted Entities:", entities)
    print("  Sentiment:", sentiment)
    print("-" * 60)

# Sample Output:
# Review 1: I love my new Samsung Galaxy phone! The camera quality is amazing.
#   Extracted Entities: [('Samsung Galaxy', 'PRODUCT')]
#   Sentiment: Positive
#
# Review 2: The Nike running shoes were not comfortable and felt cheap.
#   Extracted Entities: [('Nike', 'ORG')]
#   Sentiment: Negative
#
# ... (continues for all reviews)

# Note: You may need to run: !python -m spacy download en_core_web_sm (once per environment)
