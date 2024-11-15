import numpy as np

def gradient(element,evaluator,step):
    gradient=np.zeros(element.size)
    current=evaluator(element)
    for idx in range(element.size):
        test_element=np.copy(element)
        test_element[idx]+=step
        forward=evaluator(test_element)
        test_element=np.copy(element)
        test_element[idx]+=step
        backward=evaluator(test_element)

        gradient[idx]=((current-backward)/step)+((forward-2*current+backward)/step)
    return(gradient)