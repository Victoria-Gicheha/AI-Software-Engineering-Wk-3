# MNIST Handwritten Digit Classification using CNN ===
# Goal: Build and train a CNN to achieve >95% accuracy

# 1. Import libraries ===
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


# 2. Load the MNIST dataset ===
# TensorFlow automatically downloads it the first time
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



# 3. Preprocess the data ===
# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension (needed for CNNs)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)



# 4. Build the CNN model ===
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 classes for digits 0–9
])



# 5. Compile the model ===
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)



# 6. Train the model ===
print("\nTraining the CNN model...\n")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)



# 7. Evaluate on test data ===
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")



# 8. Visualize predictions on 5 sample test images ===
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Pred: {predicted_labels[i]}\nTrue: {y_test[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# 9. (Optional) Save the model ===
model.save("mnist_cnn_model.h5")
print("\nModel saved as 'mnist_cnn_model.h5'.")
