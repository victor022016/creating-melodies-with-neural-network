import os
import json
import music21 as m21
import numpy as np
import tensorflow.keras as keras

KERN_DATASET_PATH = "deutschl/kinder"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

# las duraciones se expresan en cuartos de largo
ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5, # 8th note
    0.75,
    1.0, # quarter note
    1.5,
    2, # half note
    3,
    4 # whole note
]


def load_songs_in_kern(dataset_path):
    """Carga todas las piezas de kern en el conjunto de datos usando music21.

    :param dataset_path (str): Path to dataset
    :return songs (lista de m21 streams): Lista que contiene todas las piezas
    """
    songs = []

    # revisa todos los archivos en el conjunto de datos y cargarlos con music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # considerar solo los archivos kern
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):
    """Rutina booleana que devuelve verdad si la pieza tiene toda la duración aceptable, Falso en caso contrario.

    :param song (m21 stream):
    :param acceptable_durations (list): Lista de duración aceptable en un trimestre
    :return (bool):
    """
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    """Transponer canción a C maj / A min

    :param piece (m21 stream): Pieza para transponer
    :return transposed_song (m21 stream):
    """

    # obtener la clave de la canción
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimar key usando music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # obtener el intervalo para la transposición. Por ejemplo, Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transponer canción por intervalo calculado
    tranposed_song = song.transpose(interval)
    return tranposed_song


def encode_song(song, time_step=0.25):
    """Convierte una partitura en una representación musical similar a una serie temporal. Cada elemento de la lista codificada representa 'min_duration'
    cuartos de largo. Los símbolos utilizados en cada paso son: integers para MIDI notes, 'r' por representar un descanso, y '_'
    para representar notas / silencios que se transfieren a un nuevo paso de tiempo. Aquí hay una codificación de muestra:

        ["r", "_", "60", "_", "_", "_", "72" "_"]

    :param song (m21 stream): Pieza para codificar
    :param time_step (float): Duración de cada paso de tiempo en un cuarto de duración
    :return:
    """

    encoded_song = []

    for event in song.flat.notesAndRests:

        # manejar notas
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
        #
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convertir la nota / silencio en notación de series de tiempo
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # si es la primera vez que vemos una nota / silencio, codifiquémosla. De lo contrario, significa que llevamos lo mismo
            # símbolo en un nuevo paso de tiempo
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # transmitir una canción codificada a str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path):

    # load folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):

        # filtrar las canciones que tienen duraciones no aceptables
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # transponer canciones a Cmaj / Amin
        song = transpose(song)

        # codificar canciones con representación de series temporales de música
        encoded_song = encode_song(song)

        # guardar canciones en un archivo de texto
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

        if i % 10 == 0:
            print(f"Song {i} out of {len(songs)} processed")


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    """Genera un archivo que coteja todas las canciones codificadas y agrega nuevos delimitadores de piezas..

    :param dataset_path (str): Ruta a la carpeta que contiene las canciones codificadas
    :param file_dataset_path (str): Ruta al archivo para guardar canciones en un solo archivo
    :param sequence_length (int): # de los pasos de tiempo a considerar para el entrenamiento
    :return songs (str): Cadena que contiene todas las canciones en el conjunto de datos + delimitadores
    """

    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # cargar canciones codificadas y agregar delimitadores
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    # eliminar el espacio vacío del último carácter de la string
    songs = songs[:-1]

    # guardar cadena que contiene todo el conjunto de datos
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


def create_mapping(songs, mapping_path):
    """Crea un archivo json que asigna los símbolos del conjunto de datos de la canción a integers

    :param songs (str): Cuerda con todas las canciones
    :param mapping_path (str): Ruta donde guardar el mapeo
    :return:
    """
    mappings = {}

    # identificar el vocabulario
    songs = songs.split()
    vocabulary = list(set(songs))

    # crear asignaciones
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # guardar vocabulario en un archivo json
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    int_songs = []

    # asignaciones de carga
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # transformar la cadena de canciones a la lista
    songs = songs.split()

    # mapear canciones a int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_sequences(sequence_length):
    """Crea muestras de datos de entrada y salida para la formación. Cada muestra es una secuencia.

    :param sequence_length (int): Duración de cada secuencia. Con una cuantificación en semicorcheas, 64 notas equivalen a 4 compases

    :return inputs (ndarray): Training inputs
    :return targets (ndarray): Training targets
    """

    # cargar canciones y asignarlas a int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    # generar las secuencias de entrenamiento
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # codificar las secuencias one-hot
    vocabulary_size = len(set(int_songs))
    # inputs size: (# of sequences, sequence length, vocabulary size)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    print(f"There are {len(inputs)} sequences.")

    return inputs, targets


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    #inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()


