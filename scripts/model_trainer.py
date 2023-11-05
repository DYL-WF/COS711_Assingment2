import time
import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation
from torch.optim.lr_scheduler import LambdaLR
import torch # PyTorch package
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim.lr_scheduler  as lr_scheduler

import data_engine 
import logging
logging.basicConfig(format='%(message)s', filename='./logs/debug.log', level=logging.INFO)
import model as CNNModel

class Trainer():
    def __init__(self,proc_mode="cuda"):
        self.model = CNNModel.CNN_Net()

        self.proc_mode = proc_mode
        device = torch.device(proc_mode)
        self.model = self.model.to(device)

        self.lossFunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.data_engine = data_engine.data_engine("meta/Train.csv")
        self.data_engine.one_hot_encode()
        logging.info(self.model)



    def Train(self, numEpochs=5, trainingSplit=0.7):
        logging.info("#==================+-------------------------- TRAINING MODEL --------------------------------+==================#")
        tic = time.perf_counter()
        lossesTotal = []
        lossesEpoch = []
        accuracyEpoch = []
        numCases = self.data_engine.get_count()
        numCases = 1000


        trainingCount = round( numCases*trainingSplit )
        validationCount = round( numCases*(1-trainingSplit) )
        for epoch in range(numEpochs):
            lr= self.optimizer.param_groups[0]['lr']
            epochAverageLoss = 0
            correct = 0
            epochtic = time.perf_counter()
            # Training
            for i in range(trainingCount):
                logging.info("")

                img_data, meta_data, target = self.data_engine.get_input(i,self.proc_mode)

                x = self.model.forward(img_data,meta_data)

                if(self.proc_mode=="cuda"):
                    y = torch.Tensor([[target]]).cuda()
                else: 
                    y = torch.Tensor([[target]])

                loss = torch.sqrt( self.lossFunction(x,y) )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lossesTotal.append(loss.item())
                epochAverageLoss += loss.item()

                logging.info(f"Epoch: {epoch+1} Training Case: {i+1}\n   Target: {round(y.item()*100)}, Prediction: {round(x.item()*100,0)}, Loss: {str(round(loss.item(),10))} \n")

            # Average Loss
            lossesEpoch.append(epochAverageLoss/trainingCount)

            # Validation
            for i in range(trainingCount,numCases):

                img_data, meta_data, target = self.data_engine.get_input(i, self.proc_mode)

                x = round(100 * self.model.forward(img_data,meta_data).item() )
                y = round(100 * torch.Tensor([[target]]).item() )

                if(x==y):
                    correct+=1

                logging.info(f"Epoch: {epoch+1} Validation Case: {i+1}\n   Target: {y}, Prediction: {x} \n")

            accuracy = round(100 * correct / validationCount ,2)
            accuracyEpoch.append(accuracy)
            self.scheduler.step()

            epochtoc = time.perf_counter()
            epoch_delta = epochtoc - epochtic
            logging.info(f"## Epoch: {epoch+1} \n   Time Taken: {epoch_delta:0.4f}, Accuracy: {accuracy}, Average Loss: {round(epochAverageLoss/trainingCount)} Learning Rate: {lr}\n")

            

        plt.clf()
        plt.plot(lossesEpoch)
        plt.xlabel("no. of epochs")
        plt.ylabel("Average loss per epoch")
        plt.savefig('average_loss_per_Epoch.png')

        plt.clf()
        plt.bar(range(trainingCount*numEpochs),lossesTotal, color ='maroon',width=1)
        plt.xlabel("no. of iterations")
        plt.ylabel("Loss per training case")
        plt.savefig('training_loss.png')

        plt.clf()
        plt.plot(accuracyEpoch)
        plt.xlabel("no. of epochs")
        plt.ylabel("Accuracy per epoch")
        plt.savefig('accuracy_per_Epoch.png')

        toc = time.perf_counter()
        print(f"Total Training Time: {toc - tic:0.4f}s")
    def GetModel(self):
        return self.model