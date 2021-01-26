import tensorflow as tf
from tensorflow import keras
import numpy as np
import gzip
import sys
try:
   import cPickle as pickle
except:
   import pickle

def get_dataset(training=True):
    #f = gzip.open('/Users/kurudj/Downloads/mnist.pkl.gz', 'rb')
    #if sys.version_info < (3,):
    #    data = pickle.load(f)
    #else:
    #    data = pickle.load(f, encoding='bytes')
    #(train_images, train_labels), (test_images, test_labels) = data
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    if training==False:
        return (np.array(test_images), np.array(test_labels))
    return (train_images, train_labels)

def print_stats(train_images, train_labels):
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    print(len(train_images))
    print("{}x{}".format(   len(train_images[0]),len(train_images[0])    ))
    countArr = [0 for k in range(10)]
    for j in train_labels:
        countArr[j]+=1
    for i in range(len(class_names)):
        print("{}. {} - {}".format(i,class_names[i], countArr[i]))

def build_model():
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='sparse_categorical_crossentropy')
    optimizer = 'sgd'
    optimizer = keras.optimizers.SGD(learning_rate=0.001)
    metrics = 'accuracy'
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    return model

def train_model(model, train_images, train_labels, T):
    model.fit(train_images,train_labels,epochs=T, verbose = 1)

def evaluate_model(model, test_images, test_labels, show_loss=True):
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose = 0)
    if show_loss==True:
        print("Loss: {:.4f}".format(test_loss))
    test_accuracy*=100
    formatted_accuracy = "{:.2f}".format(test_accuracy)
    print("Accuracy: {}%".format(formatted_accuracy))

def predict_label(model, test_images, index):
    arr = model.predict(test_images, verbose=0)[index]
    k = 0
    third = first = second = -sys.maxsize
    for i in range(len(arr)):
        if arr[i] > first:
            third = second
            second = first
            first = arr[i]
        elif arr[i] > second:
            third = second
            second = arr[i]
        elif arr[i] > third:
            third = arr[i]
    firstIndex = np.where(arr==first)
    secondIndex = np.where(arr==second)
    thirdIndex = np.where(arr==third)
    first*=100
    second*=100
    third*=100
    ffir = "{:.2f}".format(first)
    fsec = "{:.2f}".format(second)
    fthir = "{:.2f}".format(third)
    if firstIndex[0]==0:
        print("Zero: " + str(ffir) + "%")
    if firstIndex[0]==1:
        print("One: " + str(ffir) + "%")
    if firstIndex[0]==2:
        print("Two: " + str(ffir) + "%")
    if firstIndex[0]==3:
        print("Three: " + str(ffir) + "%")
    if firstIndex[0]==4:
        print("Four: " + str(ffir) + "%")
    if firstIndex[0]==5:
        print("Five: " + str(ffir) + "%")
    if firstIndex[0]==6:
        print("Six: " + str(ffir) + "%")
    if firstIndex[0]==7:
        print("Seven: " + str(ffir) + "%")
    if firstIndex[0]==8:
        print("Eight: " + str(ffir) + "%")
    if firstIndex[0]==9:
        print("Nine: " + str(ffir) + "%")
    if secondIndex[0]==0:
        print("Zero: " + str(fsec) + "%")
    if secondIndex[0]==1:
        print("One: " + str(fsec) + "%")
    if secondIndex[0]==2:
        print("Two: " + str(fsec) + "%")
    if secondIndex[0]==3:
        print("Three: " + str(fsec) + "%")
    if secondIndex[0]==4:
        print("Four: " + str(fsec) + "%")
    if secondIndex[0]==5:
        print("Five: " + str(fsec) + "%")
    if secondIndex[0]==6:
        print("Six: " + str(fsec) + "%")
    if secondIndex[0]==7:
        print("Seven: " + str(fsec) + "%")
    if secondIndex[0]==8:
        print("Eight: " + str(fsec) + "%")
    if secondIndex[0]==9:
        print("Nine: " + str(fsec) + "%")
    if thirdIndex[0]==0:
        print("Zero: " + str(fthir) + "%")
    if thirdIndex[0]==1:
        print("One: " + str(fthir) + "%")
    if thirdIndex[0]==2:
        print("Two: " + str(fthir) + "%")
    if thirdIndex[0]==3:
        print("Three: " + str(fthir) + "%")
    if thirdIndex[0]==4:
        print("Four: " + str(fthir) + "%")
    if thirdIndex[0]==5:
        print("Five: " + str(fthir) + "%")
    if thirdIndex[0]==6:
        print("Six: " + str(fthir) + "%")
    if thirdIndex[0]==7:
        print("Seven: " + str(fthir) + "%")
    if thirdIndex[0]==8:
        print("Eight: " + str(fthir) + "%")
    if thirdIndex[0]==9:
        print("Nine: " + str(fthir) + "%")
