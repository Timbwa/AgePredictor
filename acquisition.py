import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def data_acquisition(folder_path='\Dataset'):
    # _X represents data samples while _Y represents data labels
    # create validation sets with bottom 10% of training sets to monitor training and allow for Early Stopping
    # Test set remains unchanged since it's used for evaluating our project
    geoFeatTrainX = []
    geoFeatTrainY = []

    geoFeatValX = []
    geoFeatValY = []

    geoFeatTestX = []
    geoFeatTestY = []

    textFeatTrainX = []
    textFeatTrainY = []

    textFeatValX = []
    textFeatValY = []

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

            # split with reproducible results on each function call
            # here shuffle=2 ensures we have training data from all classes. Setting it to false will just split it,
            # top/bottom where top will have class 0 & 1 samples while bottom will have class 2 samples
            geoFeatTrainX, geoFeatValX, geoFeatTrainY, geoFeatValY = train_test_split(geoFeatTrainX, geoFeatTrainY,
                                                                                      test_size=0.2, shuffle=True,
                                                                                      random_state=1)

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

            # split with reproducible results on each function call
            textFeatTrainX, textFeatValX, textFeatTrainY, textFeatValY = train_test_split(textFeatTrainX,
                                                                                          textFeatTrainY,
                                                                                          test_size=0.2, shuffle=True,
                                                                                          random_state=1)

    return geoFeatTrainX, geoFeatTrainY, geoFeatValX, geoFeatValY, geoFeatTestX, geoFeatTestY, \
           textFeatTrainX, textFeatTrainY, textFeatValX, textFeatValY, textFeatTestX, textFeatTestY


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
    # can retrieve actual data using StandardScaler().inverse_transform(array)
    array = StandardScaler().fit_transform(array)

    # change labels to range 0-2 for training
    labels[labels == 1] = 0
    labels[labels == 2] = 1
    labels[labels == 3] = 2

    # make into 1 column matrix
    labels = labels.reshape(-1, 1)
    # convert labels to one-hot values to be used with categorical_crossentropy loss function
    # number of classes for all feature sets is 3. (+1) cause of zero indexing and we don't have class 0
    # labels = to_categorical(labels, num_classes=(3 + 1))

    return array, labels
