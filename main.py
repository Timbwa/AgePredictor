import acquisition
import os
import models as m
import numpy as np


def print_equal():
    return ' ==================================== '


def get_run_logdir(exp_name):
    root_log_dir = os.path.join(os.curdir, "training_logs", exp_name)
    import time
    run_id = time.strftime("run_%d_%m_%Y-%H_%M_%S")
    return os.path.join(root_log_dir, run_id)


def train_experiment(model: m.tf.keras.Model, epochs, learning_rate, batch_size, train_x, train_y, val_x, val_y, exp_name):
    run_logdir = get_run_logdir(exp_name)

    callbacks = [
        # save the model at the end of each epoch(save_best_only=False) or save the model with best performance on
        # validation set(save_best_only=True)
        m.tf.keras.callbacks.ModelCheckpoint('age_predictor_model.h5', save_best_only=True),
        # perform early stopping when there's no increase in performance on the validation set in (patience) epochs
        m.tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        # tensorboard callback
        m.tf.keras.callbacks.TensorBoard(run_logdir)
    ]

    """
        Compile model with:
            Optimizer: RMSprop
            Loss: Softmax(cross-entropy)
            learning-rate: (variable)1e-3
    """
    m.compile_model(model, learning_rate=learning_rate)
    """
    To view the training curves through tensorboard run the following command on terminal:

    $ tensorboard --logdir=C:\\Users\\PC\\CompVision\\AgePredictor\\training_logs --port=6006

    make sure to replace --logdir path with absolute windows path(with single '\') of training_logs after training starts
    """
    # fit the model using num of epochs and batch_size
    model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), epochs=epochs,
              batch_size=batch_size, callbacks=callbacks, verbose=True)


def check_num_neurons(feature_type: np.array, train_x, train_y, val_x, val_y, test_x, test_y):
    num_neurons = [20, 40, 60, 80, 100]

    for num_neuron in num_neurons:
        # create models with different number of layers
        print(f'{print_equal()}Experiment: Effect of the number of neurons. (Neurons = {num_neuron}){print_equal()}')
        model_no_hidden_layer = m.ModelNoHiddenLayer(num_neuron).create_model(feature_type.shape)
        model_one_hidden_layer = m.ModelOneHiddenLayer(num_neuron).create_model(feature_type.shape)
        model_two_hidden_layer = m.ModelTwoHiddenLayers(num_neuron).create_model(feature_type.shape)
        model_three_hidden_layer = m.ModelThreeHiddenLayers(num_neuron).create_model(feature_type.shape)

        models = [model_no_hidden_layer, model_one_hidden_layer, model_two_hidden_layer, model_three_hidden_layer]
        hidden_layers = 0
        folder_name = 'num_neuron'
        for model in models:
            """
            compile and train the model with constant params:
            epochs = 150
            learning_rate = 1e-3
            batch_size=64
            """
            print(f'{print_equal()} Hidden Layers: {hidden_layers}{print_equal()}')
            # print summary of the model
            print(model.summary())

            train_experiment(model, epochs=150, learning_rate=1e-3, batch_size=64, train_x=train_x,
                             train_y=train_y, val_x=val_x, val_y=val_y, exp_name=f'{folder_name}_{num_neuron}')

            # evaluate the model after training on the training set
            evaluate_model(model, test_x=test_x, test_y=test_y)
            hidden_layers += 1


def do_experiments(data):
    """
    perform experiments for each type of feature set: geoFeat, textFeat & geoTextFeat
    Experiments were done without regularization to check on overfitting
    :param data: all training, validation and testing sets
    :return:
    """
    geoFeatTrainX, geoFeatTrainY, geoFeatValX, geoFeatValY, geoFeatTestX, geoFeatTestY, \
    textFeatTrainX, textFeatTrainY, textFeatValX, textFeatValY, textFeatTestX, textFeatTestY, \
    geoTextFeatTrainX, geoTextFeatValX, geoTextFeatTestX = data

    feature_types_train_X = [geoFeatTrainX, textFeatTrainX, geoTextFeatTrainX]
    feature_types_train_Y = [geoFeatTrainY, textFeatTrainY, textFeatTrainY]
    feature_types_val_X = [geoFeatValX, textFeatValX, geoTextFeatValX]
    feature_types_val_Y = [geoFeatValY, textFeatValY, textFeatValY]
    feature_types_test_X = [geoFeatTestX, textFeatTestX, geoTextFeatTestX]
    feature_types_test_Y = [geoFeatTestY,textFeatTestY, textFeatTestY]

    feat_num = 0
    for index, feature_type in enumerate(feature_types_train_X):
        print(f'{print_equal()} Feature Type: {feat_num}{print_equal()}')

        # search for number of neurons keeping other hyper-parameters constant
        check_num_neurons(feature_type, feature_types_train_X[index], feature_types_train_Y[index],
                          feature_types_val_X[index], feature_types_val_Y[index], feature_types_test_X[index],
                          feature_types_test_Y[index])


def evaluate_model(model: m.tf.keras.Model, test_x, test_y):
    # evaluate the model
    print(f'{print_equal()}Evaluation{print_equal()}')
    score = model.evaluate(x=test_x, y=test_y)
    print(f'Accuracy: {score[1]}')
    print(f'{print_equal()}')


def main():
    # do acquisition
    geoFeatTrainX, geoFeatTrainY, \
    geoFeatValX, geoFeatValY, \
    geoFeatTestX, geoFeatTestY, \
    textFeatTrainX, textFeatTrainY, \
    textFeatValX, textFeatValY, \
    textFeatTestX, textFeatTestY = acquisition.data_acquisition()

    # concatenate geoFeat with textFeat, no need to concatenate label arrays. Can use either geoFeatTrainY or
    # textFeatTrainY as labels for training
    geoTextFeatTrainX = np.concatenate((geoFeatTrainX, textFeatTrainX), axis=1)
    geoTextFeatValX = np.concatenate((geoFeatValX, textFeatValX), axis=1)
    geoTextFeatTestX = np.concatenate((geoFeatTestX, textFeatTestX), axis=1)

    # do experiments
    do_experiments([geoFeatTrainX, geoFeatTrainY, geoFeatValX, geoFeatValY, geoFeatTestX, geoFeatTestY,
                    textFeatTrainX, textFeatTrainY, textFeatValX, textFeatValY, textFeatTestX, textFeatTestY,
                    geoTextFeatTrainX, geoTextFeatValX, geoTextFeatTestX])

    print('Done')


if __name__ == '__main__':
    main()
