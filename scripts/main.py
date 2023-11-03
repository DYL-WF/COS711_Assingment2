import data_engine as DataEngine;
import model as CNNModel
test = DataEngine.data_engine("meta/Train.csv")
test.one_hot_encode()

model = CNNModel.Net()

print(model.forward(test.get_input(0)[0]).item())
print(model.forward(test.get_input(1)[0]).item())
print(model.forward(test.get_input(2)[0]).item())
print(model.forward(test.get_input(3)[0]).item())
print(model.forward(test.get_input(4)[0]).item())
print(model.forward(test.get_input(5)[0]).item())