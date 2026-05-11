#DL 3 
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape for CNN
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# CNN Model
model = models.Sequential([
    layers.Input(shape=(28,28,1)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels)
)

# Evaluate
loss, acc = model.evaluate(test_images, test_labels)

print("Accuracy:", acc)

# Predict one image
predictions = model.predict(test_images[:1])

print("Predicted Label:", np.argmax(predictions))
print("Actual Label:", test_labels[0])

# Simple visual
plt.imshow(test_images[0].reshape(28,28), cmap='gray')
plt.title("Predicted: " + str(np.argmax(predictions)))
plt.show()
