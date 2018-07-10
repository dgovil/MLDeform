import json
import logging
import os

import numpy as np
import pandas
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.models import Sequential

DEFAULT_JOINT_COLUMNS = ['rx', 'ry', 'rz', 'rw', 'tx', 'ty', 'tz']

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def read_inputs(directory):
    """Read the input_data from the given directory"""
    input_file = os.path.join(directory, 'input_data.json')
    with open(input_file) as f:
        input_data = json.load(f)

    return input_data


def get_model(joint, verts, layers=3, activation='tanh',
              units=512, input_dim=100):
    """
    Build a training model based on the joint and vertices
    :param joint: RotateTransform values of the joint
    :param verts: Deltas of the vertices
    :param layers: The number of layers to create. A minimum of 2 is required.
    :param activation: The type of activation. Defaults to tanh
    :param units: The units per layer if not the input/output
    :param input_dim: The input dimensions of each layer that is not input/output
    :return: The model, name of the input node, the name of the output_node
    """
    model = Sequential()
    if layers < 2:
        logger.warning('A minimum of 2 layers is required')
        layers = 2

    input_name = 'input_node'
    output_name = 'output_node'
    for layer in range(layers):
        if not layer:
            model.add(Dense(units,
                            input_dim=joint.shape[1],
                            activation=activation,
                            name=input_name))
            continue

        if layer == (layers - 1):
            model.add(Dense(verts.shape[1],
                            activation='linear',
                            name=output_name))
            continue

        model.add(Dense(units,
                        input_dim=input_dim,
                        activation=activation,
                        name="dense_layer_%s" % layer))

    output_node = model.output.name
    input_node = '%s_input:0' % input_name
    return model, input_node, output_node


def make_plot(history, output, show=True):
    """
    Build a model from the trained keras model
    :param history: The output of the trained model
    :param output: Where to save the plot
    :param show: Whether to show the plot
    :return: The plot object
    """
    import matplotlib.pyplot as plt

    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.ylabel('mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(output)
    if show:
        plt.show()
    return plt

def normalize_features(df):
    """
    Normalize a given pandas dataframe using the min and max values.
    This puts the cells in the range 0-1. NaNs are treated as 0
    :param df: The pandas dataframe
    :return: Normalized Dataframe, a list of max values, a list of min values
    """
    df_max = df.max()
    df_min = df.min()
    df_norm = (df-df_min)/(df_max-df_min)
    df_norm = df_norm.fillna(0)
    return df_norm, df_max.values.tolist(), df_min.values.tolist()


def train(input_directory, normalize=False,
          rate=0.001, epochs=200, split=0.3, batch_size=None,
          show=True, plot=True, activation='tanh',
          units=512, input_dim=100, layers=3):
    """
    Train the model from written data
    :param input_directory: The path to the directory where the data was written to
    :param normalize: Whether to normalize inputs and outputs
    :param rate: The learning rate. Lower rates are more accurate but slower.
    :param epochs: The number of epochs to train for. Higher is more accurate but slower and there are diminishing returns.
    :param split: The training/testing split. Defaults to 0.3 for 70% training 30% test.
    :param batch_size: The batch size to train with.
    :param show: Show the plot after each model is done training.
    :param plot: Whether to generate a plot for each training model.
    :param activation: What kind of activation to use. Defaults to tanh
    :param units: What units to use for intermediate layers.
    :param input_dim: Input dimensions to use for intermediate layers.
    :param layers: The number of layers to use. A minimum of 2 is enforced.
    :return: The path to the output json file
    """

    # Read the data from disk.
    input_data = read_inputs(input_directory)
    csv_files = input_data.get('csv_files', [])
    joint_columns = input_data.get('input_fields', DEFAULT_JOINT_COLUMNS)

    for i, csv_file in enumerate(csv_files):
        # Prepare the filesystem to write
        file_name, _ext = os.path.splitext(os.path.basename(csv_file))
        export_directory = os.path.join(input_directory, file_name)
        if not os.path.exists(export_directory):
            os.mkdir(export_directory)

        logger.info('Training for %s', file_name)
        # Read the csv of vert deltas to a pandas dataframe.
        df = pandas.read_csv(csv_file)
        df = df.drop_duplicates(joint_columns)
        if not df.shape[0] or df.shape[1] <= len(joint_columns):
            input_data.setdefault('models', []).append(None)
            continue

        # Shuffle the data and split it into input and output data.
        df.reindex(np.random.permutation(df.index))
        joints = df.iloc[:, :len(joint_columns)]
        verts = df.iloc[:, len(joint_columns):]

        # Split the joint transform values.
        rot = joints.iloc[:, :-3]
        trans = joints.iloc[:, -3: ]

        # Normalize the data or use None as defaults
        verts_max, verts_min, trans_max, trans_min = None, None, None, None
        if normalize:
            logger.debug('Normalizing values')
            verts, verts_max, verts_min = normalize_features(verts)
            trans, trans_max, trans_min = normalize_features(trans)

        # Remerge the joint transform values.
        joints = pandas.concat([rot, trans], axis=1)

        # Start making the model.
        with tf.Session(graph=tf.Graph()) as session:
            # Create a model.
            model, input_name, output_name = get_model(
                joints, verts, layers=layers, units=units,
                input_dim=input_dim, activation=activation
            )

            # Generate the optimizer and train the model
            adam = keras.optimizers.Adam(lr=rate)
            model.compile(loss='mse', optimizer=adam, metrics=['mse'])
            history = model.fit(joints, verts,
                                epochs=epochs,
                                validation_split=split,
                                batch_size=batch_size)

            # Show the plots
            plot_image = None
            if plot:
                plot_image = os.path.join(export_directory, '%s.png' % file_name)
                make_plot(history, plot_image, show=show)

            export_path = os.path.join(export_directory, file_name)

            # Save the keras model as a tensorflow model
            saver = tf.train.Saver(save_relative_paths=True)
            saver.save(session, export_path)

            # Store the data in a dict
            model_data = {
                'root': export_directory,
                'meta': export_path + '.meta',
                'input': input_name,
                'output': output_name,
                'plot': plot_image,
                'normalized': normalize,
                'trans_min':trans_min,
                'trans_max':trans_max,
                'verts_min':verts_min,
                'verts_max':verts_max
            }

        # Then write this dict to the master dict
        input_data.setdefault('models', []).append(model_data)

    # Finally write this out to disk.
    output_data = os.path.join(input_directory, 'output_data.json')
    with open(output_data, 'w') as f:
        json.dump(input_data, f)

    return output_data


if __name__ == '__main__':
    training_data = '/Users/dhruv/Library/Preferences/Autodesk/maya/training/'
    print(train(training_data, layers=3, show=True, plot=True, normalize=True))
