import acquisition
import os
import models as m


def get_run_logdir():
    root_log_dir = os.path.join(os.curdir, "training_logs")
    import time
    run_id = time.strftime("run_%d_%m_%Y-%H_%M_%S")
    return os.path.join(root_log_dir, run_id)


def main():
    # do acquisition
    geoFeatTrainX, geoFeatTrainY, \
    geoFeatTestX, geoFeatTestY, \
    textFeatTrainX, textFeatTrainY, \
    textFeatTestX, textFeatTestY = acquisition.data_acquisition()

    # create models with hidden layer(s) of num_neurons each
    num_neurons = 30
    model_no_hidden_layer = m.ModelNoHiddenLayer(num_neurons).create_model(geoFeatTrainX.shape)
    model_one_hidden_layer = m.ModelOneHiddenLayer(num_neurons).create_model(geoFeatTrainX.shape)
    model_two_hidden_layer = m.ModelTwoHiddenLayers(num_neurons).create_model(geoFeatTrainX.shape)
    model_three_hidden_layer = m.ModelThreeHiddenLayers(num_neurons).create_model(geoFeatTrainX.shape)

    """
    Compile models with:
        Optimizer: RMSprop
        Loss: Softmax(cross-entropy)
        learning-rate: (variable)1e-3
    """
    learning_rate = 1e-3
    m.compile_model(model_no_hidden_layer, learning_rate=learning_rate)
    m.compile_model(model_one_hidden_layer, learning_rate=learning_rate)
    m.compile_model(model_two_hidden_layer, learning_rate=learning_rate)
    m.compile_model(model_three_hidden_layer, learning_rate=learning_rate)

    """
    To view the training curves through tensorboard run the following command on terminal:
    
    $ tensorboard --logdir=C:\\Users\\PC\\CompVision\\AgePredictor\\training_logs --port=6006
    
    make sure to replace --logdir path with absolute windows path(with single '\') of training_logs after training starts
    """
    run_logdir = get_run_logdir()

    """
        Train the models with:
            batch_size
            epochs
            callbacks: for saving model at intermediary epochs
        """
    callbacks = [
        # save the model at the end of each epoch
        m.tf.keras.callbacks.ModelCheckpoint('age_predictor_model.h5'),
        # tensorboard callback
        m.tf.keras.callbacks.TensorBoard(run_logdir)
    ]
    epochs = 150
    batch_size = 256
    # train the models
    model_one_hidden_layer.fit(x=geoFeatTrainX,
                               y=geoFeatTrainY,
                               epochs=epochs,
                               batch_size=batch_size,
                               callbacks=callbacks,
                               verbose=True)

    # evaluate the model
    print('Evaluating...')
    score = model_one_hidden_layer.evaluate(x=geoFeatTestX, y=geoFeatTestY)
    print(f'Accuracy: {score[1]}')

    print('Done')


if __name__ == '__main__':
    main()
