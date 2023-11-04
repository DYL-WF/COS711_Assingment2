from matplotlib import pyplot as plt
import data_engine as DataEngine;
import model as CNNModel
import logging
import torch

logging.basicConfig( filename='./logs/debug.log', level=logging.DEBUG)
logging.FileHandler(filename='./logs/debug.log',mode='w')

test = DataEngine.data_engine("meta/Train.csv")
test.one_hot_encode()

model = CNNModel.CNN_Net()

# print(model.forward(test.get_input(0)[0],test.get_input(0)[1]))
# print(model.forward(test.get_input(1)[0]).item())
# print(model.forward(test.get_input(2)[0]).item())
# print(model.forward(test.get_input(3)[0]).item())
# print(model.forward(test.get_input(4)[0]).item())
# print(model.forward(test.get_input(5)[0]).item())

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

logging.info("#==================+-------------------------- TRAINING MODEL --------------------------------+==================#")
logging.debug("### MODEL ###")
logging.debug(model)
lossesTotal = []
lossesEpoch = []
numCases = 26000
for epoch in range(5):
    epochAverageLoss = 0
    for i in range(numCases):
        logging.info("")
        logging.info("Epoch: " + str(epoch)+ " Training case "+str(i+1)+" of " + str(numCases))

        img_data, meta_data, target = test.get_input(i)
        x = model.forward(img_data,meta_data)
        y = torch.Tensor([[target]])


        loss = torch.sqrt(criterion(x, y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossesTotal.append(loss.item())
        epochAverageLoss += loss.item()
        logging.info(" Loss: "+str(loss.item()))
        logging.info("")

    lossesEpoch.append(epochAverageLoss/numCases)
    


plt.plot(lossesEpoch)
plt.xlabel("no. of iterations")
plt.ylabel("Average loss per epoch")
plt.savefig('Average_per_Epoch')

plt.plot(lossesTotal)
plt.xlabel("no. of iterations")
plt.ylabel("total loss per train case")
plt.savefig('Training Loss')