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


def exponential_decay(init_learning_rate, num_steps):
    """
    Decreases the initial learning rate by a factor of 10 every num_steps
    :param init_learning_rate: initial learning rate
    :param num_steps: number of epochs to reduce the learning rate by a factor of 10
    :return:
    """
    def exponential_decay_fn(epoch):
        new_learning_rate = init_learning_rate * 0.1 ** (epoch / num_steps)
        print(f'epoch: {epoch}\n'
              f'learning rate: {new_learning_rate}')
        return new_learning_rate
    return exponential_decay_fn


def train_experiment(model: m.tf.keras.Model, epochs, batch_size, train_x, train_y, val_x, val_y, exp_name,
                     learning_rate=None, steps=None):
    run_logdir = get_run_logdir(exp_name)

    # do learning rate experiment(use exponential decay of learning rate) if learning rate is None
    callbacks = []
    if learning_rate is not None:
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
    else:
        if steps is None:
            raise Exception('Please pass the `steps` parameter')

        initial_learning_rate = 1
        exponential_decay_ = exponential_decay(initial_learning_rate, steps)
        callbacks = [
            # save the model at the end of each epoch(save_best_only=False) or save the model with best performance on
            # validation set(save_best_only=True)
            m.tf.keras.callbacks.ModelCheckpoint('age_predictor_model.h5', save_best_only=True),
            # perform early stopping when there's no increase in performance on the validation set in (patience) epochs
            m.tf.keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True),
            # reduce the learning rate exponentially by a factor of 10
            m.tf.keras.callbacks.LearningRateScheduler(exponential_decay_),
            # tensorboard callback
            m.tf.keras.callbacks.TensorBoard(run_logdir)
        ]
        """
                Compile model with:
                    Optimizer: RMSprop
                    Loss: Softmax(cross-entropy)
                    learning-rate: intial_learning_rate of 10
            """
        m.compile_model(model, learning_rate=initial_learning_rate)
    """
    To view the training curves through tensorboard run the following command on terminal:

    $ tensorboard --logdir=C:\\Users\\PC\\CompVision\\AgePredictor\\training_logs\\experiment_name --port=6006

    make sure to replace --logdir path with absolute windows path(with single '\') of training_logs after training starts
    """
    # fit the model using num of epochs and batch_size
    model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), epochs=epochs,
              batch_size=batch_size, callbacks=callbacks, verbose=True)


def check_learning_rates(feature_type: np.array, train_x, train_y, val_x, val_y, test_x, test_y):
    # create models with different number of layers and best number of neurons found in prior experiment
    num_neuron_0 = 20
    num_neuron_1 = 20
    num_neuron_2 = 20
    num_neuron_3 = 20

    model_no_hidden_layer = m.ModelNoHiddenLayer(num_neuron_0).create_model(feature_type.shape)
    model_one_hidden_layer = m.ModelOneHiddenLayer(num_neuron_1).create_model(feature_type.shape)
    model_two_hidden_layer = m.ModelTwoHiddenLayers(num_neuron_2).create_model(feature_type.shape)
    model_three_hidden_layer = m.ModelThreeHiddenLayers(num_neuron_3).create_model(feature_type.shape)

    models = [model_no_hidden_layer, model_one_hidden_layer, model_two_hidden_layer, model_three_hidden_layer]

    print(f'{print_equal()}Experiment: Effect of the learning rate){print_equal()}')
    hidden_layers = 0
    folder_name = 'learning_rates'

    for model in models:
        """
            compile and train the model with constant params:
            epochs = 150
            batch_size=64
            
            About learning rate:
            Our Approach involves initially using a large learning rate then reduce it exponentially until 
            training stops making fast progress. This is done by using Early Stopping and an initial learning rate of
            10 that reduces by a factor of 10 every s=20 steps(epochs)
        """
        print(f'{print_equal()} Hidden Layers: {hidden_layers}{print_equal()}')
        # print summary of the model
        print(model.summary())

        # train with decaying learning rate. Learning rate reduces by a factor of 10 every 20 steps
        train_experiment(model, epochs=150, learning_rate=1, batch_size=64, train_x=train_x, train_y=train_y,
                         val_x=val_x, val_y=val_y, exp_name=f'{folder_name}_hidden_layer{hidden_layers}', steps=20)

        # evaluate the model after training on the training set
        evaluate_model(model, test_x=test_x, test_y=test_y)
        hidden_layers += 1


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
    feature_types_test_Y = [geoFeatTestY, textFeatTestY, textFeatTestY]

    feature_names = ['Geometric Features', 'Texture Features', 'Geometric-Texture Features']
    for index, feature_type in enumerate(feature_types_train_X):
        print(f'{print_equal()} Feature Type: {feature_names[index]}{print_equal()}')

        # search for number of neurons keeping other hyper-parameters constant
        check_num_neurons(feature_type, feature_types_train_X[index], feature_types_train_Y[index],
                           feature_types_val_X[index], feature_types_val_Y[index], feature_types_test_X[index],
                           feature_types_test_Y[index])

        # use decaying learning rate to find out an optimum learning rate
        # check_learning_rates(feature_type, feature_types_train_X[index], feature_types_train_Y[index],
        #                   feature_types_val_X[index], feature_types_val_Y[index], feature_types_test_X[index],
        #                   feature_types_test_Y[index])


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
