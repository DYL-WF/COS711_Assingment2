from matplotlib import pyplot as plt
import data_engine as DataEngine;
import model as CNNModel
import logging
import torch

logging.basicConfig( filename='./logs/debug.log', level=logging.INFO)
logging.FileHandler(filename='./logs/debug.log',mode='w')

import model_trainer
import model_tester
trainer = model_trainer.Trainer()
trainer.Train(numEpochs=10)

tester = model_tester.Tester(trainer.GetModel())
tester.Test()
tester.SaveModel()


