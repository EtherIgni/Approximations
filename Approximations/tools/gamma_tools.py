import numpy as np

def reshape_RM(gamma_vector, num_levels):
    assert len(gamma_vector.shape) == 1,                 "Reich Moore Evaluation Failed: gamma_vector is not 1 dimensional"
    assert gamma_vector.shape[0]   == num_levels*2,      "Reich Moore Evaluation Failed: gamma_vector has the wrong number of elements"
    gamma_matrix=np.zeros((num_levels,num_levels+1),float)
    gamma_matrix[:,0]=gamma_vector[:num_levels]
    gamma_matrix[:,1:]=np.diag(gamma_vector[num_levels:])
    return(gamma_matrix)