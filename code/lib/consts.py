import os

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

CROP_SIZE = (224, 224)
SPLITS = ['training', 'validation', 'test']

BUCKET_NAME = os.environ['BUCKET_NAME']
MODEL_PATH = "models/{model_name}"
