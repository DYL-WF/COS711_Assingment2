import data_engine as DataEngine;

test = DataEngine.data_engine("meta/Train.csv")
test.one_hot_encode()
test.get_input(0)
