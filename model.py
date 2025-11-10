# model.py - FULL VERSION (with predict_emotion)

import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

print("Step 1: TensorFlow imported successfully!")

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = 'model.h5'

# === PREPROCESS IMAGE ===
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# === PREDICT EMOTION ===
def predict_emotion(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array, verbose=0)
    emotion_idx = np.argmax(predictions)
    confidence = float(predictions[0][emotion_idx])
    return EMOTIONS[emotion_idx], confidence

# === CREATE OR LOAD MODEL ===
def create_or_load_model():
    print(f"Checking if {MODEL_PATH} exists...")
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded!")
        return model

    print("Creating new model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(7, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("MODEL SAVED: model.h5 created!")
    return model

if __name__ == "__main__":
    model = create_or_load_model()
    print("ALL DONE! You can now run: python app.py")