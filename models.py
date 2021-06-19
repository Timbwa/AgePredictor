import tensorflow as tf

# seed for weight kernel initializer
SEED = 777


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
        # output layer with softmax activation, 3 neuron for the classes
        output_layer = tf.keras.layers.Dense(3, activation='softmax',
                                             # use glorot normal cause of softmax
                                             kernel_initializer=tf.keras.initializers.glorot_normal(seed=SEED),
                                             # l2 regularization to avoid overfitting
                                             kernel_regularizer=tf.keras.regularizers.l2(),
                                             name='output_layer')(input_layer)

        # define model
        model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer], name='0_hidden_layer_model')

        return model


class ModelOneHiddenLayer(Model):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)

    def create_model(self, data_shape):
        # create Neural Architecture using Keras Functional API
        input_layer = tf.keras.Input(shape=data_shape[1:], name='input_layer')
        # one hidden Fully Connected layer with ReLu activation
        hidden_layer = tf.keras.layers.Dense(self.num_neurons, activation='relu',
                                             kernel_initializer=tf.keras.initializers.he_normal(seed=SEED),
                                             # l2 regularization to avoid overfitting
                                             kernel_regularizer=tf.keras.regularizers.l2(),
                                             name='hidden_layer')(input_layer)
        # batch normalization layer
        batch_norm_layer = tf.keras.layers.BatchNormalization(name='batch_norm_layer')(hidden_layer)
        # Fully Connected output layer with ReLu activation, 3 neuron for the classes
        output_layer = tf.keras.layers.Dense(3, activation='softmax',
                                             kernel_initializer=tf.keras.initializers.glorot_normal(seed=SEED),
                                             # l2 regularization to avoid overfitting
                                             kernel_regularizer=tf.keras.regularizers.l2(),
                                             name='output_layer')(batch_norm_layer)

        # define model's inputs and outputs
        model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer], name='1_hidden_layer_model')

        return model


class ModelTwoHiddenLayers(Model):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)

    def create_model(self, data_shape):
        # create Neural Architecture using Keras Functional API
        input_layer = tf.keras.Input(shape=data_shape[1:], name='input_layer')
        # two hidden Fully Connected layers with ReLu activation
        hidden_layer_1 = tf.keras.layers.Dense(self.num_neurons, activation='relu',
                                               kernel_initializer=tf.keras.initializers.he_normal(seed=SEED),
                                               # l2 regularization to avoid overfitting
                                               kernel_regularizer=tf.keras.regularizers.l2(),
                                               name='hidden_layer_1')(input_layer)
        # batch normalization layer
        batch_norm_layer = tf.keras.layers.BatchNormalization(name='batch_norm_layer_1')(hidden_layer_1)

        hidden_layer_2 = tf.keras.layers.Dense(self.num_neurons, activation='relu',
                                               kernel_initializer=tf.keras.initializers.he_normal(seed=SEED),
                                               # l2 regularization to avoid overfitting
                                               kernel_regularizer=tf.keras.regularizers.l2(),
                                               name='hidden_layer_2')(batch_norm_layer)

        # batch normalization layer
        batch_norm_layer_2 = tf.keras.layers.BatchNormalization(name='batch_norm_layer_2')(hidden_layer_2)

        # Fully Connected output layer with ReLu activation, 3 neuron for the classes
        output_layer = tf.keras.layers.Dense(3, activation='softmax',
                                             kernel_initializer=tf.keras.initializers.glorot_normal(seed=SEED),
                                             # l2 regularization to avoid overfitting
                                             kernel_regularizer=tf.keras.regularizers.l2(),
                                             name='output_layer')(batch_norm_layer_2)

        # define model's inputs and outputs
        model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer], name='2_hidden_layer_model')

        return model


class ModelThreeHiddenLayers(Model):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)

    def create_model(self, data_shape):
        # create Neural Architecture using Keras Functional API
        input_layer = tf.keras.Input(shape=data_shape[1:], name='input_layer')
        # three hidden Fully Connected layers with ReLu activation
        hidden_layer_1 = tf.keras.layers.Dense(self.num_neurons, activation='relu',
                                               kernel_initializer=tf.keras.initializers.he_normal(seed=SEED),
                                               # l2 regularization to avoid overfitting
                                               kernel_regularizer=tf.keras.regularizers.l2(),
                                               name='hidden_layer_1')(input_layer)
        # batch normalization layer
        batch_norm_layer = tf.keras.layers.BatchNormalization(name='batch_norm_layer_1')(hidden_layer_1)

        hidden_layer_2 = tf.keras.layers.Dense(self.num_neurons, activation='relu',
                                               kernel_initializer=tf.keras.initializers.he_normal(seed=SEED),
                                               # l2 regularization to avoid overfitting
                                               kernel_regularizer=tf.keras.regularizers.l2(),
                                               name='hidden_layer_2')(batch_norm_layer)
        # batch normalization layer
        batch_norm_layer_2 = tf.keras.layers.BatchNormalization(name='batch_norm_layer_2')(hidden_layer_2)

        hidden_layer_3 = tf.keras.layers.Dense(self.num_neurons, activation='relu',
                                               kernel_initializer=tf.keras.initializers.he_normal(seed=SEED),
                                               # l2 regularization to avoid overfitting
                                               kernel_regularizer=tf.keras.regularizers.l2(),
                                               name='hidden_layer_3')(batch_norm_layer_2)
        # batch normalization layer
        batch_norm_layer_3 = tf.keras.layers.BatchNormalization(name='batch_norm_layer_3')(hidden_layer_3)

        # Fully Connected output layer with ReLu activation, 3 neuron for the classes
        output_layer = tf.keras.layers.Dense(3, activation='softmax',
                                             kernel_initializer=tf.keras.initializers.glorot_normal(seed=SEED),
                                             # l2 regularization to avoid overfitting
                                             kernel_regularizer=tf.keras.regularizers.l2(),
                                             name='output_layer')(batch_norm_layer_3)

        # define model's inputs and outputs
        model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer], name='3_hidden_layer_model')

        return model


def compile_model(model: tf.keras.Model, learning_rate):
    """
    compiles the model with the following hyperparameters:
    optimizer: RMSprop
    loss: Softmax(Cross-entropy) -> SparseCategoricalCrossentropy for multiclass labels
    :param model: created keras model
    :param learning_rate:
    :return:
    """
    # compile model, returns None
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy())
