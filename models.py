import tensorflow as tf


class Model:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons

    def create_model(self, data_shape):
        pass


class ModelNoHiddenLayer(Model):
    # num_neurons is irrelevant here since we don't define any hidden layers
    def __init__(self, num_neurons):
        super().__init__(num_neurons)

    def create_model(self, data_shape):
        # create Neural Architecture using Keras Functional API
        input_layer = tf.keras.Input(shape=data_shape[1:], name='input_layer')
        # no hidden layer
        # output layer with ReLu activation, 1 neuron for the class
        output_layer = tf.keras.layers.Dense(1, activation='relu', name='output_layer')(input_layer)

        # define model
        model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

        # print summary of the model
        print(model.summary())

        return model


class ModelOneHiddenLayer(Model):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)

    def create_model(self, data_shape):
        # create Neural Architecture using Keras Functional API
        input_layer = tf.keras.Input(shape=data_shape[1:], name='input_layer')
        # one hidden Fully Connected layer with ReLu activation
        hidden_layer = tf.keras.layers.Dense(self.num_neurons, activation='relu', name='hidden_layer')(input_layer)
        # Fully Connected output layer with ReLu activation, 1 neuron for the class
        output_layer = tf.keras.layers.Dense(1, activation='relu', name='output_layer')(hidden_layer)

        # define model's inputs and outputs
        model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

        # print summary of the model
        print(model.summary())

        return model


class ModelTwoHiddenLayers(Model):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)

    def create_model(self, data_shape):
        # create Neural Architecture using Keras Functional API
        input_layer = tf.keras.Input(shape=data_shape[1:], name='input_layer')
        # two hidden Fully Connected layers with ReLu activation
        hidden_layer_1 = tf.keras.layers.Dense(self.num_neurons, activation='relu', name='hidden_layer_1')(input_layer)
        hidden_layer_2 = tf.keras.layers.Dense(self.num_neurons, activation='relu', name='hidden_layer_2')(hidden_layer_1)
        # Fully Connected output layer with ReLu activation, 1 neuron for the class
        output_layer = tf.keras.layers.Dense(1, activation='relu', name='output_layer')(hidden_layer_2)

        # define model's inputs and outputs
        model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

        # print summary of the model
        print(model.summary())

        return model


class ModelThreeHiddenLayers(Model):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)

    def create_model(self, data_shape):
        # create Neural Architecture using Keras Functional API
        input_layer = tf.keras.Input(shape=data_shape[1:], name='input_layer')
        # three hidden Fully Connected layers with ReLu activation
        hidden_layer_1 = tf.keras.layers.Dense(self.num_neurons, activation='relu', name='hidden_layer_1')(input_layer)
        hidden_layer_2 = tf.keras.layers.Dense(self.num_neurons, activation='relu', name='hidden_layer_2')(hidden_layer_1)
        hidden_layer_3 = tf.keras.layers.Dense(self.num_neurons, activation='relu', name='hidden_layer_3')(hidden_layer_2)
        # Fully Connected output layer with ReLu activation, 1 neuron for the class
        output_layer = tf.keras.layers.Dense(1, activation='relu', name='output_layer')(hidden_layer_3)

        # define model's inputs and outputs
        model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

        # print summary of the model
        print(model.summary())

        return model


def compile_model(model: tf.keras.Model, learning_rate):
    """
    compiles the model with the following hyperparameters:
    optimizer: RMSprop
    loss: Softmax(Cross-entropy)
    :param model: created keras model
    :param learning_rate:
    :return:
    """
    compiled_model = model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9),
                                   loss=tf.keras.losses.categorical_crossentropy())

    return compiled_model
