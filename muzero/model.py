import tensorflow as tf


class Gen_model:
    def __init__(self, input_dim, output_dim, learning_rate):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, states, targets, batch_size, verbose, epochs):
        self.model.fit(states, targets, batch_size=batch_size, verbose=verbose, epochs=epochs)

    def save(self):
        self.model.save("filepath" + ".h5")

    def load(self):
        return self.model.load("filepath" + ".h5")


class Residual_NN(Gen_model):
    def __init__(self, learning_rate, input_dim, output_dim, hidden_layers):
        Gen_model.__init__(self, learning_rate, input_dim, output_dim)
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.model = self._build_model()

    def Res(self, input_blocks, filters, kernel_size):
        x = self.conv_layer(input_blocks, filters, kernel_size)

        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size
        )(x)

        x = tf.keras.layers.BatchNormalization(dim=1)(x)
        x = tf.add([input_blocks, x])
        x = tf.keras.layers.LeakyReLU(x)
        return x

    def Conv(self, input_blocks, filters, kernel_size):
        x = tf.keras.layers.Conv2D(
            filters = filters,
            kernel_size=kernel_size
        )(input_blocks)
        x = tf.keras.layers.BatchNormalization(dim=1)(x)
        x = tf.keras.layers.LeakyReLU(x)
        return x

    def value_head(self, input_blocks, filters, kernel_size):
        x = self.Res(input_blocks, filters, kernel_size)
        x = self.Conv(x, filters, kernel_size)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(20)(x)
        x = tf.keras.layers.LeakyReLU(x)
        value = tf.keras.layers.Dense(1)(x)
        return value

    def policy_head(self, input_blocks, filters, kernel_size):
        x = self.Res(input_blocks, filters, kernel_size)
        x = self.Conv(x, filters, kernel_size)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(42)(x)
        return x