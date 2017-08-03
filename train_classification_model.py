# Base imports
from __future__ import print_function, division

import os
from os.path import join as pj

from config import config

# Set up GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']

import numpy as np
np.random.seed = 0  # for reproducibility


# DL imports
from tensorflow.python.client import device_lib
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Model
from keras import metrics
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau


# Local imports
import models
from experiments import Experiment


if __name__ == '__main__':
    # Setup experiment
    experiment = Experiment('./classification_experiments')
    experiment.add_dir('checkpoints')
    experiment.add_dir('tensorboard')

    # Print information about GPUs
    print(device_lib.list_local_devices())

    # Load model
    model_name = config['MODEL_NAME']
    print('Loading model {}...'.format(model_name))

    if model_name in models.name_to_model:
        model_build_function = models.name_to_model[model_name]
        model = model_build_function(
                config['IMAGE_HEIGHT'], config['IMAGE_WIDTH'],
                config['N_CHANNELS'], config['N_CLASSES'],
                lr=config['LEARNING_RATE']
        )
    else:
        raise Exception('Unknown model name: {}'.format(model_name))

    # Create batch generators

    # Train
    train_idg = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        channel_shift_range=0.1,
        horizontal_flip=True
    )

    train_generator = train_idg.flow_from_directory(
        config['TRAIN_DATA_DIR'],
        target_size=(config['IMAGE_HEIGHT'], config['IMAGE_WIDTH']),
        batch_size=config['BATCH_SIZE'],
        class_mode='categorical'
    )

    # Val
    val_idg = ImageDataGenerator(
        rescale=1 / 255
    )

    val_generator = val_idg.flow_from_directory(
        config['VAL_DATA_DIR'],
        target_size=(config['IMAGE_HEIGHT'], config['IMAGE_WIDTH']),
        batch_size=config['BATCH_SIZE'],
        class_mode='categorical'
    )


    # Setup callbacks
    callbacks = []

    # ModelCheckpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        pj(experiment.dirs['checkpoints'], 'checkpoint-{epoch:04d}.hdf5'),
        verbose=1,
        period=config['CHECKPOINTS_PERIOD']
    )
    callbacks.append(model_checkpoint_callback)

    # ReduceLROnPlateau callback
    reduce_lr_on_plateau_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config['LR_FACTOR'],
        patience=config['LR_PATIENCE'],
        min_lr=config['MIN_LR'],
        cooldown=config['COOLDOWN']
    )

    callbacks.append(reduce_lr_on_plateau_callback)

    # TensorBoard callback
    tensorboard_callback = TensorBoard(
        log_dir=experiment.dirs['tensorboard'],
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        batch_size=1
    )

    callbacks.append(tensorboard_callback)

    # Dump config
    experiment.save_json('config.json', config)

    # Train model
    model.fit_generator(train_generator,
                        steps_per_epoch=config['STEPS_PER_EPOCH'], epochs=config['EPOCHS'],
                        validation_data=val_generator, validation_steps=config['VALIDATION_STEPS'],
                        callbacks=callbacks, verbose=1)
