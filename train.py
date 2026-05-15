import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

# ================= CONFIG =================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

DATASET_PATH = "dataset"

# ================= DATA AUGMENTATION =================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,

    validation_split=0.2,

    rotation_range=20,
    zoom_range=0.2,

    width_shift_range=0.2,
    height_shift_range=0.2,

    horizontal_flip=True,

    brightness_range=[0.8, 1.2],

    fill_mode="nearest"
)

# ================= TRAIN DATA =================
train_data = datagen.flow_from_directory(
    DATASET_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True
)

# ================= VALIDATION DATA =================
val_data = datagen.flow_from_directory(
    DATASET_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False
)

# ================= SAVE LABELS =================
labels = list(train_data.class_indices.keys())

with open("labels.json", "w") as f:
    json.dump(labels, f)

print("✅ Labels saved successfully!")

# ================= LOAD RESNET50 =================
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# ================= FREEZE MOST LAYERS =================
for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

# ================= CUSTOM CLASSIFIER =================
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(256, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(128, activation="relu")(x)

x = Dropout(0.3)(x)

output = Dense(
    len(labels),
    activation="softmax"
)(x)

# ================= FINAL MODEL =================
model = Model(
    inputs=base_model.input,
    outputs=output
)

# ================= COMPILE =================
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)

# ================= CALLBACKS =================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=0.000001
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True
)

callbacks = [
    early_stop,
    reduce_lr,
    checkpoint
]

# ================= TRAIN =================
history = model.fit(
    train_data,

    validation_data=val_data,

    epochs=EPOCHS,

    callbacks=callbacks
)

# ================= SAVE MODEL =================
model.save("hybrid_model.h5")

print("✅ Model trained successfully!")
print("✅ Model saved as hybrid_model.h5")

# ================= FINAL ACCURACY =================
train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]

print(f"Training Accuracy : {train_acc * 100:.2f}%")
print(f"Validation Accuracy : {val_acc * 100:.2f}%")
