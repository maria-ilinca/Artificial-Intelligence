import numpy as np
from sklearn.kernel_ridge import KernelRidge

test_data = np.load('test_data.npy')
train_data_full = np.load('train_data.npy')
train_labels_full = np.load('train_labels.npy')

mock_train_data = train_data_full[:200]
mock_train_labels = train_labels_full[:200]

mock_val_data = train_data_full[200:]
mock_val_labels = train_labels_full[200:]


def ngram_similarity(s1, s2, n=8):
    nr = 0
    for i in range(len(s1)-n+1):
        ngram = s1[i:i+n]
        if ngram in s2:
            nr += 1

    return nr


class KNNClassifier:
    def __init__(self, train_strings, train_labels, num_neighbors=3):
        self.train_strings = train_strings
        self.train_labels = train_labels
        self.num_neighbors = num_neighbors

    def classify_string(self, test_string):
        differences = [ngram_similarity(train_string, test_string) for train_string in self.train_strings]

        sort_indices = np.argsort(differences)[::-1]
        nearest_indices = sort_indices[:self.num_neighbors]
        nearest_labels = self.train_labels[nearest_indices]

        return np.sign(np.sum(nearest_labels))

    def classify_strings(self, test_strings):
        predicted_labels = np.zeros(test_strings.shape[0])

        for i in range(test_strings.shape[0]):
            predicted_labels[i] = self.classify_string(test_strings[i])

        return predicted_labels

    def accuracy_score(self, true_labels, predicted_labels):
        return np.mean(true_labels == predicted_labels)


# Create a KNN classifier
knn = KNNClassifier(mock_train_data, mock_train_labels, num_neighbors=11)

# Predict the labels of the test data
predicted_labels = knn.classify_strings(mock_val_data)

# Compute accuracy
accuracy = knn.accuracy_score(mock_val_labels, predicted_labels)
print(11, 'Accuracy: ', accuracy)

def kernel_training_matrix():
    n = train_data_full.shape[0]
    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            K[i, j] = ngram_similarity(train_data_full[i], train_data_full[j])

    return K

def kernel_test_matrix():
    n = test_data.shape[0]
    m = train_data_full.shape[0]
    K = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = ngram_similarity(test_data[i], train_data_full[j])

    return K


def kernel_test_mock_matrix():
    n = mock_val_data.shape[0]
    m = train_data_full[0]
    K = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            K[i, j] = ngram_similarity(mock_val_data[i], mock_train_data[j])

    return K


print(kernel_training_matrix())
print(kernel_test_matrix())
print(kernel_test_mock_matrix())

for alpha in [0.1, 1, 10, 100, 1000]:
    krr = KernelRidge(alpha=alpha, kernel='precomputed')
    krr.fit(kernel_training_matrix(), train_labels_full)
    predicted_labels = krr.predict(kernel_test_mock_matrix())
    print(alpha, 'Accuracy: ', knn.accuracy_score(mock_val_labels, predicted_labels))