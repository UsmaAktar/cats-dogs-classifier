import tensorflow as tf
from keras import layers, models
from keras.preprocessing import image_dataset_from_directory

# ---------------------------
# Parameters
# ---------------------------
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "data"

# ---------------------------
# Load datasets
# ---------------------------
train_ds = image_dataset_from_directory(
    DATA_DIR + "/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = image_dataset_from_directory(
    DATA_DIR + "/validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# ---------------------------
# Performance optimization
# ---------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ---------------------------
# Build CNN model
# ---------------------------
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # Binary classification
])

# ---------------------------
# Compile
# ---------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------
# Train
# ---------------------------
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ---------------------------
# Save model
# ---------------------------
model.save("cats_dogs_model.keras")

print("✅ Model saved as cats_dogs_model.keras")
