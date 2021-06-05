import numpy as np
import os


def data_acquisition(folder_path='\Dataset'):

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

        if(filename == 'IrisGeometicFeatures_TestingSet.txt'):

            print("Doing IrisGeometicFeatures_TestingSet.txt stuff")

            file = open(full_set_path + '\\' +filename, "r")

            geoFeatTestX = file.read().splitlines()

            file.close()

            geoFeatTestX  = geoFeatTestX[10:]

            geoFeatTestX, geoFeatTestY = process_format(geoFeatTestX, 1)



        elif(filename == 'IrisGeometicFeatures_TrainingSet.txt'):

            print("Doing IrisGeometicFeatures_TrainingSet.txt stuff")

            file = open(full_set_path + '\\' + filename, "r")

            geoFeatTrainX = file.read().splitlines()

            file.close()

            geoFeatTrainX = geoFeatTrainX[10:]

            geoFeatTrainX, geoFeatTrainY = process_format(geoFeatTrainX, 1)

        elif(filename == 'IrisTextureFeatures_TestingSet.txt'):

            print("Doing IrisTextureFeatures_TestingSet.txt stuff")

            file = open(full_set_path + '\\' + filename, "r")

            textFeatTestX = file.read().splitlines()

            file.close()

            textFeatTestX = textFeatTestX[9605:]

            textFeatTestX, textFeatTestY = process_format(textFeatTestX, 2)

        elif (filename == 'IrisTextureFeatures_TrainingSet.txt'):

            file = open(full_set_path + '\\' + filename, "r")

            print("Doing IrisTextureFeatures_TrainingSet.txt stuff")

            textFeatTrainX = file.read().splitlines()

            file.close()

            textFeatTrainX = textFeatTrainX[9605:]

            textFeatTrainX, textFeatTrainY = process_format(textFeatTrainX, 2)


    #print('done')


#Function for each .txt file, so that it's array data is put in the correct format

def process_format(array,mode):

    labels = []

    if(mode == 1):

        for i in range(len(array)):

            array[i] = array[i].split(",")

            labels.append(array[i][-1])

            array[i] = array[i][:len(array[i])-1]



        array = np.array(array, dtype = np.float32)
        labels = np.array(labels, dtype = np.float32)

        labels = labels.reshape(-1, 1)

        return array, labels


    if(mode == 2):

        for i in range(len(array)):
            array[i] = array[i].split(",")

            labels.append(array[i][-1])

            array[i] = array[i][:len(array[i]) - 1]

        array = np.array(array, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        labels = labels.reshape(-1, 1)

        return array, labels