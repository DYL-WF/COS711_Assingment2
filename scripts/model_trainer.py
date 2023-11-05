import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer

import data_engine 
import logging
logging.basicConfig(format='%(message)s', filename='./logs/debug.log', level=logging.INFO)
import model as CNNModel

class Trainer():
    def __init__(self, learningRate=0.001):
        self.model = CNNModel.CNN_Net()
        self.lossFunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)   

        self.data_engine = data_engine.data_engine("meta/Train.csv")
        self.data_engine.one_hot_encode()



    def Train(self, numEpochs=5, trainingSplit=0.7):
        logging.info("#==================+-------------------------- TRAINING MODEL --------------------------------+==================#")

        lossesTotal = []
        lossesEpoch = []
        accuracyEpoch = []
        numCases = self.data_engine.get_count()
        # numCases = 100

        trainingCount = round( numCases*trainingSplit )
        validationCount = round( numCases*(1-trainingSplit) )
        for epoch in range(numEpochs):
            epochAverageLoss = 0
            correct = 0

            # Training
            for i in range(trainingCount):
                logging.info("")

                img_data, meta_data, target = self.data_engine.get_input(i)

                x = self.model.forward(img_data,meta_data)
                y = torch.Tensor([[target]])

                loss = torch.sqrt(self.lossFunction(x,y))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lossesTotal.append(loss.item())
                epochAverageLoss += loss.item()

                logging.info(f"Epoch: {epoch+1} Training Case: {i+1}\n   Target: {round(y.item()*100)}, Prediction: {round(x.item()*100)}, Loss: {str(loss.item())} \n")

            # Average Loss
            lossesEpoch.append(epochAverageLoss/trainingCount)

            # Validation
            for i in range(trainingCount,numCases):

                img_data, meta_data, target = self.data_engine.get_input(i)

                x = round(100 * self.model.forward(img_data,meta_data).item() )
                y = round(100 * torch.Tensor([[target]]).item() )

                if(x==y):
                    correct+=1

                logging.info(f"Epoch: {epoch} Validation Case: {i+1}\n   Target: {y}, Prediction: {x} \n")

            accuracy = 100 * correct / validationCount 
            accuracyEpoch.append(accuracy)
            logging.info(f"## Epoch: {epoch} \n   Accuracy: {accuracy}, Average Loss: {epochAverageLoss} \n")

            

        plt.clf()
        plt.plot(lossesEpoch)
        plt.xlabel("no. of epochs")
        plt.ylabel("Average loss per epoch")
        plt.savefig('average_loss_per_Epoch', format="png", dpi=1200)

        plt.clf()
        plt.bar(range(trainingCount*numEpochs),lossesTotal, color ='maroon',width=1)
        plt.xlabel("no. of iterations")
        plt.ylabel("Loss per training case")
        plt.savefig('training_loss', format="png", dpi=1200)

        plt.clf()
        plt.plot(accuracyEpoch)
        plt.xlabel("no. of epochs")
        plt.ylabel("Accuracy per epoch")
        plt.savefig('accuracy_per_Epoch', format="png", dpi=1200)

    def GetModel(self):
        return self.model