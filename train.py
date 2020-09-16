import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 25
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"


def build_model(output_units, num_units, loss, learning_rate):
    """Construye y compila el modelo

    :param output_units (int): Número de unidades de salida
    :param num_units (list of int): Número de unidades en capas ocultas
    :param loss (str): Tipo de función de pérdida a utilizar
    :param learning_rate (float): Tasa de aprendizaje para aplicar

    :return model (tf model): Donde ocurre la magia:D
    """

    # crear la arquitectura del modelo
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compilar modelo
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """Entrenar y guardar modelo TF.

    :param output_units (int): Número de unidades de salida
    :param num_units (list of int): Número de unidades en capas ocultas
    :param loss (str): Tipo de función de pérdida a utilizar
    :param learning_rate (float): Tasa de aprendizaje para aplicar
    """

    # generar las secuencias de entrenamiento
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # construir la red
    model = build_model(output_units, num_units, loss, learning_rate)

    # entrenar al modelo
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # guardar el modelo
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()