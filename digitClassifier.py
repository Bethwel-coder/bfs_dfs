# Imports
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
from PIL import Image
 
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
 
 
# 1. Load & preprocess MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., tf.newaxis] / 255.0
x_test  = x_test [..., tf.newaxis] / 255.0
 
 
# 2. Build model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
 
    # Data augmentation — only active during training
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.2),
 
    # Feature extraction
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
 
    # Classification head — Dropout after the dense layer it regularises
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),          # 50 % of neurons dropped during each training step
    layers.Dense(10, activation="softmax"),
])
 
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",   # correct loss for integer labels
    metrics=["accuracy"],
)
 
model.summary()
 
 
# 3. Train 
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
 
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
)
 
 
# 4. Evaluate 
loss, acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")
 
 
# 5. Save model
model.save("mnist_model.keras")
print("Model saved to mnist_model.keras")
 
 
# 6. Image helpers 
def load_image(filepath: str) -> np.ndarray:
    """Read an image file and binarise it to match MNIST conventions."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {filepath}")
 
    # Adaptive threshold handles uneven lighting and ruled lines
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,   
        C=10,
    )
 
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0
 
 
def center_digit(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr > 0.1)
    if len(coords) == 0:
        return arr                       
 
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped = arr[y0:y1 + 1, x0:x1 + 1]
 
    h, w = cropped.shape
    size = max(h, w) + 8                  
    padded = np.zeros((size, size), dtype=np.float32)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = cropped
 
    return cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
 
 
# 7. Load, preprocess & predict custom images
image_files = [
    "zero.jpeg", "one.jpeg", "two.jpeg", "three.jpeg", "four.jpeg",
    "five.jpeg", "six.jpeg", "seven.jpeg", "eight.jpeg", "nine.jpeg",
]
 
processed_images = []
for file in image_files:
    arr = load_image(file)
    arr = center_digit(arr)        
    processed_images.append(arr)
 
# Stack into a batch of shape (N, 28, 28, 1) and run inference
batch = np.stack(processed_images)[..., np.newaxis]
predictions = model.predict(batch)
predicted_labels = np.argmax(predictions, axis=1)
confidences     = np.max(predictions, axis=1)
 
 
# 8. Visualise: original vs. preprocessed vs. prediction
fig, axes = plt.subplots(3, len(image_files), figsize=(20, 6))
 
for i, file in enumerate(image_files):
    # Row 0 — original image
    original = Image.open(file).convert("L").resize((28, 28))
    axes[0, i].imshow(original, cmap="gray")
    axes[0, i].set_title(file.split(".")[0], fontsize=6)
    axes[0, i].axis("off")
 
    # Row 1 — what the model receives after preprocessing
    axes[1, i].imshow(processed_images[i], cmap="gray")
    axes[1, i].axis("off")
 
    # Row 2 — prediction result
    axes[2, i].axis("off")
    colour = "green" if predicted_labels[i] == i else "red"
    axes[2, i].set_title(
        f"Pred: {predicted_labels[i]}\n{confidences[i]:.0%}",
        fontsize=7,
        color=colour,
    )
 
axes[0, 0].set_ylabel("Original",    fontsize=8)
axes[1, 0].set_ylabel("Model input", fontsize=8)
axes[2, 0].set_ylabel("Prediction",  fontsize=8)
 
plt.tight_layout()
plt.savefig("predictions.png", dpi=150)  
plt.show()
 
print("\nPredictions:")
for file, label, conf in zip(image_files, predicted_labels, confidences):
    print(f"  {file:<12}  →  {label}  ({conf:.1%})")
 