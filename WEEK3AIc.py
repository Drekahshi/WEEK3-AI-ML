
# Buggy TensorFlow MNIST script (for troubleshooting challenge)

import tensorflow as tf
from tensorflow.keras import layers, models

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# [BUG] Data normalization mistake: Should divide by 255.0, not 256.0
x_train = x_train / 256.0
x_test = x_test / 256.0

# [BUG] Missing channel dimension for Conv2D
# [BUG] Model expects (28, 28, 1) but gets (28, 28)
# (No expansion of dims)

# [BUG] Model output: Should be 10 units (for 10 digits)
# [BUG] Using categorical_crossentropy, but labels are integers (should be one-hot)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28)),  # [BUG]
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(8, activation='softmax')  # [BUG: should be 10]
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # [BUG: y_train is not one-hot]
              metrics=['accuracy'])

# [BUG] No one-hot encoding of y_train/y_test
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)



# Fixed TensorFlow MNIST script

import tensorflow as tf
from tensorflow.keras import layers, models

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Correct normalization
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension for Conv2D
x_train = x_train[..., None]  # shape (N, 28, 28, 1)
x_test = x_test[..., None]

# One-hot encode labels for categorical_crossentropy
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# Correct model input shape and output units
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 digits
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train_cat, epochs=3, batch_size=128, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print("Test accuracy:", test_acc)