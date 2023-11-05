from matplotlib import pyplot as plt
import data_engine as DataEngine;
import model as CNNModel
import logging
import torch

logging.basicConfig( filename='./logs/debug.log', level=logging.INFO)
logging.FileHandler(filename='./logs/debug.log',mode='w')

import model_trainer
import model_tester
trainerGPU = model_trainer.Trainer(proc_mode="cuda")
trainerGPU.Train(numEpochs=5)

# trainerCPU = model_trainer.Trainer(proc_mode="cpu")
# trainerCPU.Train(numEpochs=10)

tester = model_tester.Tester(trainerGPU.GetModel(),proc_mode="cuda")
tester.Test()
tester.SaveModel()


