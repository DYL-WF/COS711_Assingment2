
import torch
import data_engine 
import logging

import model as CNNModel
logging.basicConfig(format='%(message)s', filename='./logs/debug.log', level=logging.INFO)

class Tester():
    def __init__(self, model,proc_mode="cuda"):
        self.model = model  
        self.proc_mode = proc_mode
        self.data_engine = data_engine.data_engine("meta/Test.csv","test")
        self.data_engine.one_hot_encode()
       
    def SaveModel(self):
        torch.save(self.model.state_dict(), "model/submission")

    def LoadModel(self):
        self.model = torch.load("model/submission")
        
    def Test(self):
        logging.info("#==================+-------------------------- TESTING MODEL --------------------------------+==================#")

        numCases = self.data_engine.get_count()
        f = open("submission", "w")

        for i in range(numCases):
            img_data, meta_data, id = self.data_engine.get_test_input(i,self.proc_mode)
            x = round(self.model.forward(img_data,meta_data).item()*100)

            f.write(f"{id} {x}\n")
            logging.info(f"Testing Case {i+1}\n   Prediction: {x} \n")

        f.close()
            