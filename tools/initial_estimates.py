import numpy as np

def diag_approx_guess(gamma_matrix,L_matrix):
    approx=np.zeros(gamma_matrix.shape)
    approx[:,0]=gamma_matrix[:,0]
    for idr in range(gamma_matrix.shape[0]):
        weighted_average=np.mean(np.divide(gamma_matrix[idr,1:]@L_matrix[:,1:,1:]@gamma_matrix[idr,1:].T,L_matrix[:,idr+1,idr+1]))
        approx[idr,idr+1]=weighted_average
    return(approx)

def single_value_approx(gamma_matrix):
    U,S,Vh=np.linalg.svd(gamma_matrix)
    U=U[:,0]
    S=S[0]
    Vh=Vh[0,:]
    result=np.zeros((np.max([U.size,Vh.size,3]),3))
    result[:U.size,0]=U.T
    result[0,1]=S
    result[:Vh.size,2]=Vh
    return(result)

#print(single_value_approx(sg.gamma_matrix))