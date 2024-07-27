import json
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
import tensorflow as tf
from keras import layers, models
from tensorflow.keras.models import Sequential

data_dir = pathlib.Path('classification_image')

batch_size = 32
img_height = 40
img_width = 40


def classify_predict(model, img_bgr):
    img_bgr = cv2.resize(img_bgr, (img_width, img_height))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    images = np.expand_dims(img_rgb, axis=0)
    predictions = model.predict_on_batch(images)
    exp_x = [2.7 ** x for x in predictions[0]]
    percent_score_list = [round(x * 100 / sum(exp_x)) for x in exp_x]
    highest_score_index = np.argmax(predictions[0])  # 3
    highest_score_percent = percent_score_list[highest_score_index]
    return highest_score_index, highest_score_percent


def train_classification():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=(img_width, img_height),
        batch_size=batch_size)

    class_names = train_ds.class_names
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)

    model = Sequential([
        layers.Input(shape=(img_width, img_height, 3)),
        layers.RandomFlip("horizontal"),
        # layers.RandomRotation(0.1),
        # layers.RandomZoom(0.1),
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=5)

    model.summary()

    epochs = 130
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        # callbacks=[early_stopping, lr_scheduler]
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Adjust epochs_range based on the actual number of epochs run
    epochs = len(acc)
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    model.save('model.h5')


def predict_test():
    classify_model = models.load_model('model.h5')

    error_n = 0
    data_dir = pathlib.Path('classification_image')

    for file_name in os.listdir(os.path.join(data_dir, '0none')):
        file_path = os.path.join(data_dir, '0none', file_name)
        image = cv2.imread(file_path)
        index, percent = classify_predict(classify_model, image)
        if index != 0:
            error_n += 1
            print(f'0none {index}, {percent}')
            print(file_path)
            resize_image = cv2.resize(image, (img_width, img_height))
            cv2.imshow('0none', cv2.resize(resize_image, (0, 0), fx=6, fy=6))
            cv2.waitKey(0)

    for file_name in os.listdir(os.path.join(data_dir, '1enemy')):
        file_path = os.path.join(data_dir, '1enemy', file_name)
        image = cv2.imread(file_path)
        index, percent = classify_predict(classify_model, image)
        if index != 1:
            error_n += 1
            print(f'1enemy {index}, {percent}')
            print(file_path)
            resize_image = cv2.resize(image, (img_width, img_height))
            cv2.imshow('1enemy', cv2.resize(resize_image, (0, 0), fx=6, fy=6))
            cv2.waitKey(0)


if __name__ == '__main__':
    train_classification()
    predict_test()
