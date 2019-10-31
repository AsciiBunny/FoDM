import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import exercise3 as exercise

data = pd.read_excel("HW3Atrain.xlsx")
data_valid = pd.read_excel("HW3Avalidate.xlsx")
data_valid0 = pd.read_excel("HW3Avalidate0.xlsx")
data_valid1 = pd.read_excel("HW3Avalidate1.xlsx")


def do_exercise1():
    A = exercise.NeuralNetwork()
    A.set_to_Zero()
    A.learning_phase(data, number_epochs=5000)

    np.array(A.J)
    plt.plot(A.J)
    plt.show()

    A.validation_phase(data_valid)

def do_exercise2():
    B = exercise.NeuralNetwork()
    map_info = np.zeros((11,7)) # sigma to learning rate
    sigma_array =[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    learning_rate_array= [0.0001,0.001,0.01,0.1,0.4,0.7,1]
    for sigma_i in range(len(sigma_array)):
        for learning_rate_i in range(len(learning_rate_array)):
            B.set_to_normal_distibution(sigma_array[sigma_i])
            B.learning_phase(data,number_epochs=100, learning_rate=learning_rate_array[learning_rate_i],to_print=False)
            B.validation_phase(data_valid=data_valid, to_print=False)
            map_info[sigma_i,learning_rate_i]=B.accuracy
    print(map_info)

    fig, ax = plt.subplots()
    im = plt.imshow(map_info, cmap='hot', interpolation='nearest')

    ax.set_xticks(np.arange(len(learning_rate_array)))
    ax.set_yticks(np.arange(len(sigma_array)))
    ax.set_xticklabels(learning_rate_array)
    ax.set_yticklabels(sigma_array)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    ax.set_title("Accuracy (darker is better)")
    fig.tight_layout()
    plt.show()

def do_exercise3():
    sigma_model1 = 0.2
    learning_accuracy_model1=0.001

    #initial model
    model1 = exercise.NeuralNetwork()
    model1.set_to_normal_distibution(sigma_model1)
    model1.vizualize_layers(data_valid)

    #model half way
    model1.learning_phase(data, number_epochs=50, learning_rate=learning_accuracy_model1, to_print=False)
    model1.vizualize_layers(data_valid)

    #final model
    model1.learning_phase(data, number_epochs=50, learning_rate=learning_accuracy_model1, to_print=False)
    model1.vizualize_layers(data_valid)

    sigma_model2 = 0.5
    learning_accuracy_model2 = 0.01

    # initial model
    model2 = exercise.NeuralNetwork()
    model2.set_to_normal_distibution(sigma_model2)
    model2.vizualize_layers(data_valid)

    # model half way
    model2.learning_phase(data, number_epochs=50, learning_rate=learning_accuracy_model2, to_print=False)
    model2.vizualize_layers(data_valid)

    # final model
    model2.learning_phase(data, number_epochs=50, learning_rate=learning_accuracy_model2, to_print=False)
    model2.vizualize_layers(data_valid)

do_exercise3()