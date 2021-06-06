import numpy as np
import os
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler


def data_acquisition(folder_path='\Dataset'):
    # _X represents data samples while _Y represents data one-hot class labels
    geoFeatTrainX = []
    geoFeatTrainY = []

    geoFeatTestX = []
    geoFeatTestY = []

    textFeatTrainX = []
    textFeatTrainY = []

    textFeatTestX = []
    textFeatTestY = []

    full_set_path = os.getcwd() + folder_path

    for filename in os.listdir(full_set_path):
        if filename == 'IrisGeometicFeatures_TestingSet.txt':
            print("Reading IrisGeometicFeatures_TestingSet.txt ...")

            file = open(full_set_path + '\\' + filename, "r")
            geoFeatTestX = file.read().splitlines()
            file.close()

            geoFeatTestX = geoFeatTestX[10:]

            # split into data and labels
            geoFeatTestX, geoFeatTestY = process_format(geoFeatTestX)

        elif filename == 'IrisGeometicFeatures_TrainingSet.txt':
            print("Reading IrisGeometicFeatures_TrainingSet.txt ...")

            file = open(full_set_path + '\\' + filename, "r")
            geoFeatTrainX = file.read().splitlines()
            file.close()

            geoFeatTrainX = geoFeatTrainX[10:]

            # split into data and labels
            geoFeatTrainX, geoFeatTrainY = process_format(geoFeatTrainX)

        elif filename == 'IrisTextureFeatures_TestingSet.txt':
            print("Reading IrisTextureFeatures_TestingSet.txt ...")

            file = open(full_set_path + '\\' + filename, "r")
            textFeatTestX = file.read().splitlines()
            file.close()

            textFeatTestX = textFeatTestX[9605:]

            # split into data and labels
            textFeatTestX, textFeatTestY = process_format(textFeatTestX)

        elif filename == 'IrisTextureFeatures_TrainingSet.txt':
            print("Reading IrisTextureFeatures_TrainingSet.txt ...")

            file = open(full_set_path + '\\' + filename, "r")
            textFeatTrainX = file.read().splitlines()
            file.close()

            textFeatTrainX = textFeatTrainX[9605:]

            # split into data and labels
            textFeatTrainX, textFeatTrainY = process_format(textFeatTrainX)

    return geoFeatTrainX, geoFeatTrainY, geoFeatTestX, geoFeatTestY, textFeatTrainX, textFeatTrainY, textFeatTestX, \
           textFeatTestY


def process_format(array):
    """
    Function for each .txt file, so that it's array data is put in the correct format
    :param array:
    :param mode:
    :return:
    """
    labels = []

    for i in range(len(array)):
        array[i] = array[i].split(",")
        labels.append(array[i][-1])
        array[i] = array[i][:len(array[i]) - 1]

    array = np.array(array, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # zero-center array data to the range [-1, 1], with std of 1
    # can retrieve actual data using scaler.inverse_transform(array)
    array = StandardScaler().fit_transform(array)

    # convert labels to one-hot values to be used with categorical_crossentropy loss function
    # number of classes for all feature sets is 3. (+1) cause of zero indexing and we don't have class 0
    # labels = to_categorical(labels, num_classes=(3 + 1))

    return array, labels
