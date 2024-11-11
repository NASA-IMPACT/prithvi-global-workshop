from __future__ import absolute_import

import argparse
import boto3
import os
import os.path as osp
import time
import time
import yaml
import logging

from lib.trainer import Trainer
from lib.utils import download_data, assumed_role_session, upload_model_artifacts

ROLE_ARN = os.environ.get('ROLE_ARN')
ROLE_NAME = os.environ.get('ROLE_NAME')

logger = logging.getLogger(__name__)


def train():
    config_file = os.environ.get('CONFIG_FILE')
    print(f'\n config file: {config_file}')

    print(f"Environment variables: {os.environ}")

    # download and prepare data for training:
    for split in ['training', 'validation', 'test', 'configs', 'models']:
        download_data(os.environ.get('S3_URL'), split)

    with open(config_file) as config:
        config = yaml.safe_load(config)

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    model_path = f"/opt/ml/code/{os.environ['VERSION']}/{os.environ['EVENT_TYPE']}/"
    os.makedirs(model_path)
    os.makedirs('/opt/ml/code/predicted')

    log_file = osp.join(config['logging']['checkpoint_dir'], f'{timestamp}.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)

    trainer = Trainer(config_file)
    logger.info(trainer.model)
    trainer.train()

    session = assumed_role_session()
    s3_connection = session.resource('s3')
    upload_model_artifacts(s3_connection, trainer.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # This is a way to pass additional arguments when running as a script
    # and use sagemaker-containers defaults to set their values when not specified.
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args = parser.parse_args()

    train()
