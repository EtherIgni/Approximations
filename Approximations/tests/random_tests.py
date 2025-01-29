import numpy as np
import matplotlib.pyplot as plt
import os 
# print(os.getcwd())
# from Approximations.models import gamma_SVD_model
# from Approximations.tools import numeric_evaluators

# def test_eval(x):
#     return(np.sum(np.power(x,2)))

# print(test_eval(np.array([2,2])))
# print(numeric_evaluators.gradient(np.array([2,2],float),test_eval,0.0000000001))

# print("a b ".split(" ")[:-1])

# with open("Approximations/test_file.txt","w") as file:
#     file.write("")

print(np.max(np.linalg.svd(np.array([[3,6],[7,2]]))[1]))