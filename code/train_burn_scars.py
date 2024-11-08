from lib.trainer import Trainer

def train():
    trainer = Trainer('../config/burn_scars.yaml')
    trainer.train()
