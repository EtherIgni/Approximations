import numpy as np

def numeric_gradient(element,evaluator,step):
    gradient=np.zeros(element.size)
    current=evaluator(element)
    for idx in range(element.size):
        test_element=np.copy(element)
        test_element[idx]+=step
        forward=evaluator(test_element)
        gradient[idx]=(forward-current)/step
    return(gradient)