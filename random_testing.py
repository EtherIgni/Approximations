import numpy as np
import matplotlib.pyplot as plt
from tools import initial_estimates
import copy
# matrix=np.zeros((5,4))
# for idr in range(matrix.shape[0]):
#     for idc in range(matrix.shape[1]):
#         element=matrix[idr,idc]=idr+idc
# print(matrix)

# A=np.array([[1,2],[3,4]])
# B=np.tile(A.T,[2,1,1])
# print(B)

# C=np.array([[1,1,0],[1,0,1]])
# print(np.ndarray.flatten(C))
# indexes=np.where(np.ndarray.flatten(C)==1)[0]
# print(indexes)
# D=np.array([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]])
# print(D[:,indexes])

# x=np.linspace(1,4,4)
# y=np.linspace(1,4,4)
# # x_new=np.repeat(x,y.size)
# # print(x_new)
# # y_new=np.tile(y,x.size)
# # print(y_new)
# x,y=np.meshgrid(x,y)
# z=np.empty(y.shape)
# for row in range(y.shape[0]):
#     for col in range(y.shape[1]):
#         z[row,col]=y[row,col]+x[row,col]
# fig = plt.figure()
# ax = plt.axes(projection ='3d')
# ax.plot_surface(x,y,z)
# plt.show()

# w=1
# for a in range(10):
#     if a=="a":
#         break
#     print(a)
#     w+=1
# else:
#     print("Q")

# A=np.array([[1,2,3],[4,5,6],[7,8,9]],float)
# print(initial_estimates.single_value_approx(A))

# B=[np.array([1,3,7,9],float)[:,None],
#    np.array([0.1],float),
#    np.array([2,5,6],float)[None]]
# print(B[0]@B[1][:,None]@B[2])

# for y in range(100):
#    print(y,np.roots([1,(2*y)-100,(y**2)-y]))

# num_dice=4
# die_type=6

# results_1=np.zeros(num_dice*2*die_type-num_dice*2+1)
# results_2=[]
# def roll_dice(num_dice_left,previous_rolls,num_dice):
#    if num_dice_left==1:
#       for roll in range(die_type):
#          number=roll
#          for previous_roll in previous_rolls:
#             number+=previous_roll
#          results_1[number]+=1
#          results_2.append((number+num_dice))
#    else:
#       for roll in range(die_type):
#          new_rolls=copy.copy(previous_rolls)
#          new_rolls.append(roll)
#          roll_dice(num_dice_left-1,new_rolls,num_dice)

# roll_dice(num_dice*2,[],num_dice*2)

# results_2=np.array(results_2)
# results_2_double=results_2
# result_1_double=results_1

# results_1=np.zeros(num_dice*die_type-num_dice+1)
# results_2=[]

# roll_dice(num_dice,[],num_dice)

# results_2=np.array(results_2)
# results_2_mult=results_2*2
# result_1_mult=results_1

# print(results_2_double)
# print(results_2_mult)

# print(np.mean(results_2_double))
# print(np.mean(results_2_mult))
# plt.ylim((0,0.4))
# plt.xlim((num_dice*2,num_dice*2*die_type))
# plt.grid(True,zorder=0)
# plt.hist(results_2_double,bins=np.arange(num_dice*2-0.5,num_dice*2*die_type+1.5,1),bottom=0.2,density=True,zorder=1)
# plt.vlines(np.mean(results_2_double),0,0.4,"k",zorder=2)
# plt.hist(results_2_mult,bins=np.arange(num_dice-1,num_dice*2*die_type+2,2),density=True,zorder=1)
# plt.show()

# arr=[1,2,3,4]
# print(np.arange(1,len(arr)+1,1))

# for x in range(1000):
#     for y in range(1000):
#         for z in range(1000):
#             calc=(x+1)/(y+z+2)+(y+1)/(x+z+2)+(z+1)/(x+y+2)
#             if(calc==4):
#                 print(x,y,z)

# suffixes=["","k","M","G"]
# resonance_gap=float(3E5)
# print(resonance_gap)
# rg_exponent=int(np.floor(np.floor(np.log10(resonance_gap))/3))
# rg_mantisaa=resonance_gap/(10**(np.floor(np.log10(resonance_gap))-np.floor(np.log10(resonance_gap))%3))
# problem_name=format(rg_mantisaa,".0f")+" "+suffixes[rg_exponent]+"ev"
# print(problem_name)
# print(rg_exponent)
# print(np.floor(np.floor(np.log10(resonance_gap))/3))

# def test_function(gamma_matrix):
#     return(np.sum(np.power(gamma_matrix,2)))
# def gradient(x):
#     return(3*np.power(x,2))
# def hessian(x):
#     return(np.array([[6*x[0],0],[0,6*x[1]]]))

# # running=np.array([5,2])
# # print(running," ",0)
# # for idx in range(50):
# #     running=running-np.linalg.inv(hessian(running))@gradient(running)
# #     print(running," ",idx+1)



# def derivative_numeric_gm(gamma_matrix,evaluator,step_size_relative):
#         result=evaluator(gamma_matrix)
#         gradient=np.zeros(gamma_matrix.size,float)
#         col_count=gamma_matrix.shape[0]+1
#         for idx in range(gradient.size):
#             gamma_der=np.copy(gamma_matrix)
#             abs_step=step_size_relative*gamma_der[int(np.floor(idx/col_count)),idx%col_count]
#             gamma_der[int(np.floor(idx/col_count)),idx%col_count]+=abs_step
#             gradient[idx]=(evaluator(gamma_der)-result)/abs_step
#         return(gradient)

# gamma_matrix=np.array([[1,1,1],[1,1,1]],float)
# step_size_relative=float(1E-12)

# def gradient_high_wrapper(element,step_size_relative):
#     def gradient_low_wrapper(gamma_matrix):
#         return(derivative_numeric_gm(gamma_matrix,test_function,step_size_relative)[element])
#     return(gradient_low_wrapper)

# gradient=derivative_numeric_gm(gamma_matrix,test_function,step_size_relative)
# hessian=np.zeros((gamma_matrix.size,gamma_matrix.size),float)
# for idx in range(gamma_matrix.size):
#     hessian[:,idx]=derivative_numeric_gm(gamma_matrix,gradient_high_wrapper(idx,step_size_relative),float(1E-2))
# print(gradient)
# print(hessian)
# print(-np.linalg.inv(hessian)@gradient)










# def calc_A(gamma_matrix,L):
#     return(np.array([[3,4],[5,6]])-gamma_matrix@L@gamma_matrix.T)
# def calc_A_inv(gamma_matrix,L):
#     return(np.linalg.inv(calc_A(gamma_matrix,L)))
# def calc_U(gamma_matrix,L,omega,P):
#     return(2j*omega@P@gamma_matrix.T@calc_A_inv(gamma_matrix,L)@gamma_matrix@P@omega)
# def calc_xs(gamma_matrix,L,omega,P):
#     xs=np.zeros((gamma_matrix.shape[1],L.shape[0]))
#     U_matrix=calc_U(gamma_matrix,L,omega,P)
#     xs[0,:]=3E-10*(1-2*U_matrix[:,0,0].real+(np.conjugate(U_matrix[:,0,0])*U_matrix[:,0,0]).real)
#     for i in range(1,gamma_matrix.shape[1]):
#         xs[i,:]=3E-10*(np.conjugate(U_matrix[:,0,i])*U_matrix[:,0,i]).real
#     return(xs)
# def calc_error(gamma_matrix,L,omega,P,trues):
#     xs=calc_xs(gamma_matrix,L,omega,P)
#     error=np.sum(np.power(np.sum(trues-xs,0),2)+np.power(np.sum(trues[1:]-xs[1:],0),2))
#     return(error)





# data_Types=[np.float64,np.complex128]
# step_size=np.array([0.000001],data_Types[0])
# energy_length=10

# gamma_matrix_test=np.array([[1,1,1],[1,1,1]],data_Types[0])

# trues=np.array([1E10,2E10,3E10],data_Types[0])
# trues=np.repeat(trues[:,np.newaxis],energy_length,1)
# L=np.array([[6,0,0],[0,7,0],[0,0,8]],data_Types[0])
# L=np.repeat(L[np.newaxis,:,:],energy_length,0)
# omega=np.array([[1,2,3],[4,5,6],[7,8,9]])
# omega=np.repeat(omega[np.newaxis,:,:],energy_length,0)
# P=np.array([[11,12,13],[14,15,16],[17,18,19]])
# P=np.repeat(P[np.newaxis,:,:],energy_length,0)

# A_inv=calc_A_inv(gamma_matrix_test,L)
# U_matrix=calc_U(gamma_matrix_test,L,omega,P)
# xs=calc_xs(gamma_matrix_test,L,omega,P)
# error=calc_error(gamma_matrix_test,L,omega,P,trues)
# iterable_mapping=np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]],int)


# def calc_hessian_and_gradient(gamma_matrix,iterable_mapping,energy_length,A_inv,U_matrix,xs):
#     gamma_shape=gamma_matrix_test.shape
#     num_independent=iterable_mapping.shape[0]
    
#     gamma_gradient=np.zeros((6,2,3),data_Types[0])
#     for i in range(num_independent):
#         gamma_gradient[i,iterable_mapping[i,0],iterable_mapping[i,1]]=1
    
    
#     A_gradient=np.zeros((num_independent,energy_length,gamma_shape[0],gamma_shape[0]),data_Types[0])
#     for i in range(num_independent):
#         A_gradient[i]=-gamma_gradient[i]@L@gamma_matrix.T-gamma_matrix@L@gamma_gradient[i].T
    
#     A_hessian=np.zeros((num_independent,num_independent,energy_length,gamma_shape[0],gamma_shape[0]),data_Types[0])
#     for i in range(num_independent):
#         for j in range(num_independent):
#             if(iterable_mapping[i,1]==iterable_mapping[j,1]):
#                 A_hessian[i,j]=-gamma_gradient[i]@L@gamma_gradient[j].T-gamma_gradient[j]@L@gamma_gradient[i].T
#             else:
#                 A_hessian[i,j]=0


#     A_inv_gradient=np.zeros((num_independent,energy_length,gamma_shape[0],gamma_shape[0]),data_Types[0])
#     for i in range(num_independent):
#         A_inv_gradient[i]=-A_inv@A_gradient[i]@A_inv
    
#     A_inv_hessian=np.zeros((num_independent,num_independent,energy_length,gamma_shape[0],gamma_shape[0]),data_Types[0])
#     for i in range(num_independent):
#         for j in range(num_independent):
#             A_inv_hessian[i,j]=A_inv@A_gradient[i]@A_inv@A_gradient[j]@A_inv-A_inv@A_hessian[i,j]@A_inv+A_inv@A_gradient[j]@A_inv@A_gradient[i]@A_inv


#     U_gradient=np.zeros((num_independent,energy_length,gamma_shape[1],gamma_shape[1]),data_Types[1])
#     for i in range(num_independent):
#         U_gradient[i]=2j*omega@P@(gamma_gradient[i].T@A_inv@gamma_matrix+
#                                 gamma_matrix.T@A_inv_gradient[i]@gamma_matrix+
#                                 gamma_matrix.T@A_inv@gamma_gradient[i])@P@omega
#     U_hessian=np.zeros((num_independent,num_independent,energy_length,gamma_shape[1],gamma_shape[1]),data_Types[1])
#     for i in range(num_independent):
#         for j in range(num_independent):
#             interstage =gamma_gradient[i].T@A_inv_gradient[j]@gamma_matrix
#             interstage+=gamma_gradient[i].T@A_inv@gamma_gradient[j]
#             interstage+=gamma_gradient[j].T@A_inv_gradient[i]@gamma_matrix
#             interstage+=gamma_matrix.T@A_inv_hessian[i,j]@gamma_matrix
#             interstage+=gamma_matrix.T@A_inv_gradient[i]@gamma_gradient[j]
#             interstage+=gamma_gradient[j].T@A_inv@gamma_gradient[i]
#             interstage+=gamma_matrix.T@A_inv_gradient[j]@gamma_gradient[i]
#             U_hessian[i,j]=2j*omega@P@interstage@P@omega


#     xs_gradient=np.zeros((num_independent,gamma_shape[1],energy_length),data_Types[0])
#     for i in range(num_independent):
#         xs_gradient[i,0]=3E-10*(-2*U_gradient[i,:,0,0].real+(np.conjugate(U_gradient[i,:,0,0])*U_matrix[:,0,0]+np.conjugate(U_matrix[:,0,0])*U_gradient[i,:,0,0]).real)
#         for j in range(1,gamma_shape[1]):
#             xs_gradient[i,j]=3E-10*(np.conjugate(U_gradient[i,:,0,j])*U_matrix[:,0,j]+np.conjugate(U_matrix[:,0,j])*U_gradient[i,:,0,j]).real
            
#     xs_hessian=np.zeros((num_independent,num_independent,gamma_shape[1],energy_length),data_Types[0])
#     for i in range(num_independent):
#         for j in range(num_independent):
#             xs_hessian[i,j,0]=3E-10*(-2*U_hessian[i,j,:,0,0].real+
#                                 (np.conjugate(U_hessian[i,j,:,0,0]) *U_matrix[:,0,0]+
#                                 np.conjugate(U_gradient[i,:,0,0])  *U_gradient[j,:,0,0]+
#                                 np.conjugate(U_gradient[j,:,0,0])  *U_gradient[i,:,0,0]+
#                                 np.conjugate(U_matrix[:,0,0])      *U_hessian[i,j,:,0,0]).real)
#             for k in range(1,gamma_shape[1]):
#                 xs_hessian[i,j,k]=3E-10*(np.conjugate(U_hessian[i,j,:,0,k]) * U_matrix[:,0,k]+
#                                     np.conjugate(U_gradient[i,:,0,k])  * U_gradient[j,:,0,k]+
#                                     np.conjugate(U_gradient[j,:,0,k])  * U_gradient[i,:,0,k]+
#                                     np.conjugate(U_matrix[:,0,k])      * U_hessian[i,j,:,0,k]).real


#     error_gradient=np.zeros(num_independent,data_Types[0])
#     for i in range(num_independent):
#         error_gradient[i]=np.sum(2*np.sum(trues-xs,0)*np.sum(-xs_gradient[i],0)+2*np.sum(trues[1:]-xs[1:],0)*np.sum(-xs_gradient[i,1:],0))
        
#     error_hessian=np.zeros((num_independent,num_independent),data_Types[0])
#     for i in range(num_independent):
#         for j in range(num_independent):
#             error_hessian[i,j]=np.sum(2*np.sum(-xs_gradient[j],0)    * np.sum(-xs_gradient[i],0)+
#                                     2*np.sum(trues-xs,0)           * np.sum(-xs_hessian[i,j],0)+
#                                     2*np.sum(-xs_gradient[j,1:],0) * np.sum(-xs_gradient[i,1:],0)+
#                                     2*np.sum(trues[1:]-xs[1:],0)   * np.sum(-xs_hessian[i,j,1:],0))
    
    
#     return(error_gradient,error_hessian)


# error_gradient,error_hessian=calc_hessian_and_gradient(gamma_matrix_test,iterable_mapping,energy_length,A_inv,U_matrix,xs)
# print(error_gradient)
# print(error_hessian)
# print(" ")
# print(-np.linalg.inv(error_hessian)@error_gradient)







p=6
A=np.eye(p,p+1,1)
A[:,0]=1
print(A)
print(A.nonzero())