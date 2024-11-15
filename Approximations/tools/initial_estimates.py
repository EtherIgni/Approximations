import numpy as np

def reich_moore_guess(gamma_matrix):
    approx=np.zeros(gamma_matrix.shape)
    num_resonances=gamma_matrix.shape[0]
    approx=np.zeros(num_resonances*2)
    approx[:num_resonances]=gamma_matrix[:,0]
    approx[num_resonances:num_resonances*2]=np.diag(gamma_matrix[:,1:])
    return(approx)

def gamma_SVD_approx(gamma_matrix):
    num_resonances=gamma_matrix.shape[0]
    approx=np.zeros(num_resonances*3)
    approx[:num_resonances]=gamma_matrix[:,0]
    u,s,vh=np.linalg.svd(gamma_matrix[:,1:])
    approx[num_resonances:num_resonances*2]=u[:,0]*s[0]
    approx[num_resonances*2:num_resonances*3]=vh[0,:]
    return(approx)