import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:
    """Una clase que envuelve el modelo LSTM y ofrece utilidades para generar melodías."""

    def __init__(self, model_path="model.h5"):
        """Constructor que inicializa el modelo TensorFlow"""

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Genera una melodía usando el modelo DL y devuelve un archivo midi.

        :param seed (str): Semilla de melodía con la notación utilizada para codificar el conjunto de datos
        :param num_steps (int): Número de pasos a generar
        :param max_sequence_len (int): Número máximo de pasos en la semilla a considerar para la generación
        :param temperature (float): Float en intervalo [0, 1]. Los números más cercanos a 0 hacen que el modelo sea más determinista.
            Un número más cercano a 1 hace que la generación sea más impredecible.

        :return melody (list of str): Lista con símbolos que representan una melodía
        """

        # crear semilla con símbolos de inicio
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # mapear semilla a int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # limitar la semilla a max_sequence_length
            seed = seed[-max_sequence_length:]

            # codificar en caliente la semilla
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # Hace una predicción
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # actualizar semilla
            seed.append(output_int)

            # mapear int a nuestra codificación
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # comprobar si estamos al final de una melodía
            if output_symbol == "/":
                break

            # actualizar melodía
            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self, probabilites, temperature):
        """Muestra un índice de una matriz de probabilidad aplicando softmax usando temperatura

        :param predictions (nd.array): Matriz que contiene probabilidades para cada una de las posibles salidas.
        :param temperature (float): Float in interval [0, 1]. Los números más cercanos a 0 hacen que el modelo sea más determinista.
            A number closer to 1 makes the generation more unpredictable.

        :return index (int): Símbolo de salida seleccionado
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index


    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        """Convierte la melodía en un archivo MIDI

        :param melody (list of str):
        :param min_duration (float): Duración de cada paso de tiempo en un cuarto de duración
        :param file_name (str): Nombre del archivo midi
        :return:
        """

        # crear una secuencia de music21
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # analizar todos los símbolos de la melodía y crear objetos de nota / silencio
        for i, symbol in enumerate(melody):

            # Manejar caso en el que tengamos una nota / resto
            if symbol != "_" or i + 1 == len(melody):

                # asegúrese de que estamos tratando con la nota / el resto más allá del primero
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # 
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # 
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # restablecer el contador de pasos
                    step_counter = 1

                start_symbol = symbol

            # Manejar caso en el que tengamos un cartel de prolongación"_"
            else:
                step_counter += 1

        # escribe la secuencia m21 en un archivo midi
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
    print(melody)
    mg.save_melody(melody)




















