config = dict()

# Model
config['MODEL_NAME'] = 'green_light_net_v1'

# Image sizes
config['BASE_SIZE'] = 30
config['IMAGE_HEIGHT'], config['IMAGE_WIDTH'] = 9 * config['BASE_SIZE'], 16 * config['BASE_SIZE']

config['TRAIN_DATA_DIR'] = './data/train/train_images_by_class'
config['VAL_DATA_DIR'] = './data/train/val_images_by_class'

# Input parameters
config['N_CHANNELS'] = 3
config['N_CLASSES'] = 2

# Train parameters
config['BATCH_SIZE'] = 32
config['EPOCHS'] = 100000
config['STEPS_PER_EPOCH'] = 100
config['VALIDATION_STEPS'] = 10

config['LEARNING_RATE'] = 0.001

config['CHECKPOINTS_PERIOD'] = 20

# GPU
config['CUDA_VISIBLE_DEVICES'] = '0'

# ReduceLROnPlateau
config['LR_FACTOR'] = 0.5
config['LR_PATIENCE'] = 10
config['MIN_LR'] = 0.000001
config['COOLDOWN'] = 10
