import numpy as np
import PIL as pil
from tensorflow import keras

dir = '/kaggle/input/unibuc-brain-ad/data/data/'

np_images = []
images = []
for i in range(1, 22150):
    # read the image
    image = pil.Image.open(dir + str(i).zfill(6) + '.png').convert('L')
    # convert the image to numpy array
    np_image = np.array(image)
    # append the image to the list
    np_images.append(np_image)

np_images = np.array(np_images)
# normalize the data

# reading the validation labels
val_labels = []
with open('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', 'r') as f:
    # read without the first line
    val_labels = f.readlines()[1:]
    val_labels = [x.strip() for x in val_labels]
    # the format is id,class
    val_labels = [x.split(',') for x in val_labels]
    val_labels = np.array(val_labels)
# convert into int array
# take only the second column
val_labels = val_labels[:, 1].astype(int)

train_labels = []
with open('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', 'r') as f2:
    # read without the first line
    train_labels = f2.readlines()[1:]
    train_labels = [x.strip() for x in train_labels]
    # the format is id,class
    train_labels = [x.split(',') for x in train_labels]
    train_labels = np.array(train_labels)

# take only the second column

train_labels = train_labels[:, 1].astype(int)

# save the labels into a npy file
np.save('train_labels.npy', train_labels)

# reshape the images to 164x164
np_images = np_images.reshape(np_images.shape[0], -1)
np_images = np.array([np.array(pil.Image.fromarray(np_image).resize((164, 164))) for np_image in np_images])
# #normalize the data
np_images = np_images / 255

#print(np_images.shape)

# create an augmented dataset for the training data
from keras.preprocessing.image import ImageDataGenerator
train_data = ImageDataGenerator(
    rotation_range = 10,
    width_shift_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    )

cnn_model = keras.models.Sequential([

    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(164, 164, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),
    keras.layers.BatchNormalization(),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(1, activation='sigmoid')
])

# fit the data for training
train_data.fit(np_images[:15000])
# early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# apply cnn model to the images
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_data.flow(np_images[:15000], train_labels[:15000]), epochs=100, validation_data=(np_images[15000:17000], val_labels), callbacks=[early_stopping])

# evaluate the model on the validation set
from sklearn import metrics
val_pred = cnn_model.predict(np_images[15000:17000])
print(metrics.f1_score(val_labels, np.round(val_pred).astype(int),average = None))

#confusion matrix and classification report for the validation set
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(val_labels, np.round(val_pred)))
print(classification_report(val_labels, np.round(val_pred)))

history = cnn_model.fit(np_images[:15000], train_labels[:15000], epochs=100, validation_data=(np_images[15000:17000], val_labels), callbacks=[early_stopping])
#plot the loss and accuracy for the training and validation set
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#plot the accuracy for the training and validation set
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# predict the test data
with open('submissioncnn.csv', 'w') as m:
    # The file should contain a header and have the following format:

    # id,class
    # 017001,0
    # 017002,0
    # 017003,0
    # 017004,0
    # 017005,0
    prediction = cnn_model.predict(np_images[17000:])
    m.write('id,class\n')
    for i in range(17000, len(np_images)):
        # print the id and the predicted class
        # i used np.round to round the prediction to the nearest integer
        # prediction[i - 17000][0] is the prediction for the i-th image
        # i take the first element of the array because the prediction is an array of one element
        m.write('0' + str(i + 1) + ',' + str(np.round(prediction[i - 17000][0]).astype('int')) + '\n')
        # the conversion to int is necessary because the prediction is a float

m.close()
