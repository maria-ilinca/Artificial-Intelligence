import numpy as np
import PIL as pil
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

dir = 'D:\Facultate\sem2_an_2\data\data\\'
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

labels = []
with open('D:\Facultate\sem2_an_2\data\\train_labels.txt', 'r') as f:
    # read without the first line
    labels = f.readlines()[1:]
    labels = [x.strip() for x in labels]
    # the format is id,class
    labels = [x.split(',') for x in labels]
    labels = np.array(labels)

#take only the second column
labels = labels[:,1]

# save the labels into a npy file
np.save('labels.npy', labels)

# reading the validation labels
val_labels = []
with open('D:\Facultate\sem2_an_2\data\\validation_labels.txt', 'r') as f:
    # read without the first line
    val_labels = f.readlines()[1:]
    val_labels = [x.strip() for x in val_labels]
    # the format is id,class
    val_labels = [x.split(',') for x in val_labels]
    val_labels = np.array(val_labels)
# convert into int array
val_labels = val_labels[:,1].astype(int)
#loaded_images = np.load('images.npy')

np_images = np.array(np_images)

# flatten the images
np_images = np_images.reshape(np_images.shape[0], -1)

# apply multinomial NB on the images
multinomialNB = MultinomialNB()
# take only the first 15000 images
loaded_labels = np.load('labels.npy')
multinomialNB.fit(np_images[:15000], loaded_labels)

# predict the validation labels
predicted = multinomialNB.predict(np_images[15000:17000]).astype('int')

print(f1_score(val_labels, predicted, average=None))


#confusion matrix and classification report for the validation set
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(val_labels, predicted))
print(classification_report(val_labels, predicted))

# prediction on the test set
predicted = multinomialNB.predict(np_images[17000:]).astype('int')

with open('D:\Facultate\sem2_an_2\data\\multinomial.csv', 'w') as m:
    # The file should contain a header and have the following format:

    # id,class
    # 017001,0
    # 017002,0
    # 017003,0
    # 017004,0
    # 017005,0

    m.write('id,class\n')
    for i in range(17000, len(np_images)):
        # write the id and the predicted class
        # the reshape is needed because the predict function expects a 2D array
        m.write('0' + str(i + 1) + ',' + str(multinomialNB.predict(np_images[i].reshape(1, -1))[0]) + '\n')

m.close()
