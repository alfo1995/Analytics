# -*- coding: utf-8 -*-
"""
================================
Digits Classification Exercise (8x8)
================================
A tutorial exercise regarding the use of classification techniques on
the Digits dataset.
This dataset is made up of 1797 8x8 images. Each image, is of a hand-written 
digit. In order to utilize an 8x8 figure like this, we’d have to first 
transform it into a feature vector with length 64.
k-fold cross-validation.

"""
print(__doc__)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import datasets, metrics, tree
import matplotlib.pyplot as plt

digits = datasets.load_digits()
print('Stampa delle prime 4 immagini di training')
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:8]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label) 
plt.show()

X_digits = digits.data
y_digits = digits.target

# stima del modello
clf = tree.DecisionTreeClassifier(criterion='entropy',splitter='best',max_features=None,
                                  max_depth=7,min_samples_split=10,min_samples_leaf=5)

# k-fold cross-validation --> stima errore
scores = cross_val_score(clf, X_digits, y_digits, cv=5)
# stampe
print('\n Stampa dei risultati ')
print('Tree k-fold CV-error: %f' % scores.mean())

# k-fold cross-validation --> stima Y
predicted = cross_val_predict(clf, X_digits, y_digits, cv=5)
print("Classification report for classifier %s:\n\n%s\n"
      % (clf, metrics.classification_report(y_digits, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_digits, predicted))

