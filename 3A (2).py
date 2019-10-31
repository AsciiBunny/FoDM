import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import exercise3 as exercise


data = pd.read_excel("HW3Atrain.xlsx")
A = exercise.NeuralNetwork()
A.learning_phase(data, number_epochs=1000)

np.array(A.J)
plt.plot(A.J)
plt.show()


data_valid = pd.read_excel("HW3Avalidate.xlsx")
A.validation_phase(data_valid)