import data_engine as DataEngine;

test = DataEngine.data_engine("meta/Train.csv")
test.one_hot_encode()
test.get_input(0)
test.get_input(1)
test.get_input(2)
test.get_input(3)
test.get_input(4)