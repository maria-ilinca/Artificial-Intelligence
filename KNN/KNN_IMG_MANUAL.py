import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

class KNNClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric='l2'):
        differences = np.zeros(self.train_images.shape[0])

        if metric == 'l2':
            differences = np.sqrt(np.sum(np.square(self.train_images - test_image), axis=1))
        elif metric == 'l1':
            differences = np.sum(np.abs(self.train_images - test_image), axis=1)

        sorted_indices = np.argsort(differences)
        nearest_indices = sorted_indices[:num_neighbors]
        nearest_labels = self.train_labels[nearest_indices]

        return np.argmax(np.bincount(nearest_labels.astype(int)))

    def classify_images(self, test_images, num_neighbors=3, metric='l2'):
        predicted_labels = np.zeros(test_images.shape[0])

        for i in range(test_images.shape[0]):
            predicted_labels[i] = self.classify_image(test_images[i], num_neighbors, metric)

        return predicted_labels

    def accuracy_score(self, t_labels, p_labels):
        return np.mean(t_labels == p_labels)


train_images = np.loadtxt("data/train_images.txt")
train_labels = np.loadtxt("data/train_labels.txt")
test_images = np.loadtxt("data/test_images.txt")
test_labels = np.loadtxt("data/test_labels.txt")

classifier = KNNClassifier(train_images, train_labels)
predicted_labels = classifier.classify_images(test_images, num_neighbors=3, metric='l2')
print(classifier.classify_image(train_images[0], num_neighbors=1))
print(train_labels[0])
print(classifier.accuracy_score(test_labels, predicted_labels))

# plot the accuracy for l1 and l2 metrics for neighbors in 1, 3, 5, 7, 9
for metric in ['l1', 'l2']:
    accuracies = []
    for num_neighbors in range(1, 10, 2):
        predicted_labels = classifier.classify_images(test_images, num_neighbors, metric)
        accuracies.append(classifier.accuracy_score(test_labels, predicted_labels))

    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy")
    plt.plot(range(1, 10, 2), accuracies, label=metric)
    plt.legend()
    plt.show()