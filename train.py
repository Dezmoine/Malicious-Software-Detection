!pip install --upgrade tensorflow_hub
!pip install sklearn

import os
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sklearn
from sklearn.metrics import confusion_matrix

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


from google.colab import drive
drive.mount('/content/drive')

!unzip "/content/drive/MyDrive/Projects/malware_detection/malimg_dataset.zip" -d "/content"

base_dir = os.path.join('/content', 'malimg_dataset')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

BATCH_SIZE = 32
IMAGE_SHAPE = 224

train_data = ImageDataGenerator().flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
    class_mode='categorical')

image_batch, labels_batch = next(train_data)
class_names = list(train_data.class_indices.keys())
class_names

def plots(ims, figsize=(20,30), rows=10, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = 8
    for i in range(0,32):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(list(class_names)[np.argmax(titles[i])], fontsize=12)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

plots(image_batch, titles=labels_batch)

train_image_gen = ImageDataGenerator(rescale=1.0/255)
train_data_gen = train_image_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
    class_mode='categorical')

def plotImages(images_array):
    fig, axes = plt.subplots(1, 5, figsize = (20,20))
    axes = axes.flatten()
    for img, ax in zip(images_array, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

val_image_gen = ImageDataGenerator(rescale=1.0/255)
val_data_gen = val_image_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=val_dir,
    target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
    class_mode='categorical')



test_image_gen = ImageDataGenerator(rescale=1.0/255)
test_data_gen = test_image_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=test_dir,
    target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
    class_mode='categorical')

feature_extractor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1",
    trainable=False,
    input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))

METRICS = [
    keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall')
]

num_classes = 25

model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(num_classes, activation='softmax') ])
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=METRICS)

EPOCHS = 50

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=10,
    mode='min',
    restore_best_weights=True)

info = model.fit(
    train_data_gen,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=val_data_gen)


plt.plot(info.history['loss'], label='train loss')
plt.plot(info.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.close()

plt.plot(info.history['categorical_accuracy'], label='train acc')
plt.plot(info.history['val_categorical_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.close()

test_image_batch, test_labels_batch = next(test_data_gen)

score = model.evaluate(test_data_gen, steps=BATCH_SIZE)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

train_predictions = model.predict(train_data_gen)

print(train_predictions[0])

print(tf.argmax(train_predictions[0]))
test_predictions = model.predict(test_data_gen)

print(test_predictions[0])

print(tf.argmax(test_predictions[0]))

def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_", " ").capitalize()
    plt.subplot(2, 2, n+1)
    plt.plot(history.epoch,
             history.history[metric], color=colors[0], label='Training')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8, 1])
    else:
      plt.ylim([0, 1])
    plt.legend()

plot_metrics(info)

train_predictions = model.predict(train_data_gen, batch_size=BATCH_SIZE)
test_predictions = model.predict(test_data_gen, batch_size=BATCH_SIZE)

test_labels = test_data_gen.classes
train_labels = train_data_gen.classes

results = model.evaluate(train_data_gen,
                         batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(model.metrics_names, results):
  print(name, ': ', value)

results = model.evaluate(test_data_gen,
                         batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(model.metrics_names, results):
  print(name, ': ', value)

def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5, 20])
  plt.ylim([80, 100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')

plot_roc("Train Weighted", labels_batch,
         train_predictions, color=colors[0])
plot_roc("Test Weighted", test_labels_batch, test_predictions,
         color=colors[0], linestyle='--')

plt.legend(loc='lower right')

"""## EXporting as SavedModel"""

t = time.time()
export_dir = "./{}".format(int(t))
tf.keras.models.save_model(model, export_dir)

!zip -r model.zip {export_dir}

try:
  from google.colab import files
  files.download('./model.zip')
except:
  pass
