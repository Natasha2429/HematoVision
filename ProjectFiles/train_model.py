import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # ✅ Skip corrupted images

# Paths (adjust if needed)
train_dir = "dataset/dataset2-master/dataset2-master/images/TRAIN"
test_dir = "dataset/dataset2-master/dataset2-master/images/TEST"

# Image properties
img_size = 224
batch_size = 16  # ✅ Reduced for slower machines

# Data loading
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train = train_gen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

test = test_gen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# Transfer learning using MobileNetV2
base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
pred = Dense(4, activation='softmax')(x)

model = Model(inputs=base.input, outputs=pred)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train,
    validation_data=test,
    epochs=10,
    verbose=1
)


# Save the model
model.save("Blood Cell.h5")
print("✅ Model saved as 'Blood Cell.h5'")


