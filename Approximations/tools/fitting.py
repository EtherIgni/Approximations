import numpy as np
import math
import matplotlib.pyplot as plt

def coordinate_descent(initial_values,steps,evaluator,max_num_iterations,debug):
    num_rows=initial_values.shape[0]
    num_cols=initial_values.shape[1]
    new_values=initial_values
    previous_result=evaluator(initial_values)
    for k in range(max_num_iterations):
        best_value=np.copy(new_values)
        for i in range(num_rows):
            for j in range(num_cols):
                test_matrix=np.copy(new_values)
                best_direction=0
                value_to_replace=test_matrix[i,j]
                to_beat=previous_result
                if debug: print("Run",k)
                
                stepped_value=new_values[i,j]+steps[i,j]
                test_matrix[i,j]=stepped_value
                if debug: print(test_matrix)
                if debug: print(stepped_value,"=",new_values[i,j],"+",steps[i,j])
                try:
                    test_result=evaluator(test_matrix)
                    if(test_result<to_beat):
                        best_direction=1
                        to_beat=test_result
                        value_to_replace=stepped_value
                    if debug: print("Step Up:",test_result)
                except:
                    if debug: print("error")
                    if debug: print("")
                    steps[i,j]=steps[i,j]/2
                    continue
                
                stepped_value=new_values[i,j]-steps[i,j]
                test_matrix[i,j]=stepped_value
                if debug: print(test_matrix)
                if debug: print(stepped_value,"=",new_values[i,j],"-",steps[i,j])
                try:
                    test_result=evaluator(test_matrix)
                    if(test_result<to_beat):
                        best_direction=-1
                        value_to_replace=stepped_value
                    if debug: print("Step down:",test_result)
                except:
                    if debug: print("error")
                    if debug: print("")
                    steps[i,j]=steps[i,j]/2
                    continue
                
                if debug: print(new_values)
                if debug: print("No change:",previous_result)
                if debug: print("Position: ({r},{c})".format(r=i,c=j))
                if best_direction==1:
                    best_value[i,j]=value_to_replace
                    if debug: print("up")
                elif best_direction==-1:
                    best_value[i,j]=value_to_replace
                    if debug: print("down")
                else:
                    steps[i,j]=steps[i,j]/2
                    if debug: print("half")
                if debug: print("")
                new_values=best_value
                previous_result=evaluator(new_values)
        if debug: print("-----------------------------")
        if debug: print("")
        if(np.max(steps)<float(1E-15)):
            break
    print("Converged after {k} Iterations.".format(k=k))
    print("")
    if debug: print("=============================")
    if debug: print("")
    return(new_values)

def gradient_descent_numeric(initial_values,iterable,finite_step,gradient_step,evaluator,tolerance,max_num_iterations,debug):
    num_rows=initial_values.shape[0]
    num_cols=initial_values.shape[1]
    new_values=initial_values
    converged=False
    for k in range(max_num_iterations):
        gradient=np.zeros((num_rows,num_cols))
        for idr in range(gradient.shape[0]):
            for idc in range(gradient.shape[1]):
                test_matrix=np.copy(new_values)
                test_matrix[idr,idc]=new_values[idr,idc]+finite_step
                step_forward=evaluator(test_matrix)
                test_matrix[idr,idc]=new_values[idr,idc]-finite_step
                step_backward=evaluator(test_matrix)
                gradient[idr,idc]=-(step_forward-step_backward)*iterable[idr,idc]
        if debug: print(k+1)
        if debug: print(gradient)
        if debug: print("-----------------------------")
        if debug: print("")
        new_values=new_values+gradient*(gradient_step/(finite_step*2))
        if(np.sum(np.power(gradient,2))<tolerance):
            converged=True
            break
    if converged:
        print(f"Converged after {k+1} Iterations.")
    else:
        print("Did Not Converge.")
    print("")
    if debug: print("=============================")
    if debug: print("")
    return(new_values)

def gradient_descent_half_step(initial_values,iterable,derivative,max_gradient_step,evaluator,tolerance,max_num_iterations,step_count_lim,convergance_type,debug,previous_settings=None):
    new_values=initial_values
    previous_result=evaluator(new_values)
    converged=False
    if debug>2: num_true_steps=0
    con_step_down=0
    con_no_step=0
    if(not(previous_settings==None)):
        max_gradient_step=previous_settings[0]
        con_step_down=previous_settings[1]
        con_no_step=previous_settings[2]
    for k in range(max_num_iterations[0]):
        gradient=derivative(new_values)*iterable
        if debug>3:
            x=np.linspace(0,max_gradient_step,1000)
            y=np.empty(x.shape)
            for idx,i in enumerate(x):
                test_matrix=new_values-gradient*i
                y[idx]=evaluator(test_matrix)

            plt.plot(x,y)
            plt.xscale("linear")
            plt.show()
        if debug>2: print("Iteration:",k+1)
        if debug>2: print("Gradient\n",gradient)
        if debug>2: print("Gradient Magnitude:",np.sqrt(np.sum(np.power(gradient,2))))
        current_step_size=max_gradient_step
        if debug>2: print("")
        for l in range(max_num_iterations[1]):
            if debug>2: print("Step Count:",l+1)
            if debug>2: print("Step Size:",current_step_size)
            test_matrix=new_values-gradient*current_step_size
            if debug>2: print("New Position\n",test_matrix)
            current_result=evaluator(test_matrix)
            if debug>2: print("Result to Beat:",previous_result)
            if debug>2: print("Step Result:",current_result)
            if(current_result<previous_result):
                new_values=test_matrix
                if debug>2: num_true_steps+=1
                if debug>2: print("Step")
                if debug>2: print("")
                break
            current_step_size=current_step_size/2
            if debug>2: print("")
        else:
            if debug>0: print("Failed to step Before Failsafe Triggered")
            break
        if debug>2: print("Final Position:\n",new_values)
        if debug>2: print("")
        if debug>2: print("-------------------------------------------------------------------------------------------------------------------------------------------------")
        if debug==2: print(k+1,l+1)
        if debug>1: print("")
        if(current_step_size<max_gradient_step):
            con_step_down+=1
            con_no_step=0
        else:
            con_no_step+=1
            con_step_down=0
        if(con_no_step==step_count_lim):
            max_gradient_step=max_gradient_step*2
            con_no_step=0
        if(con_step_down==step_count_lim):
            max_gradient_step=max_gradient_step/2
            con_step_down=0
        if((np.sqrt(np.sum(np.power(gradient,2)))<tolerance)and(convergance_type==0)):
            converged=True
            break
        if((current_result-previous_result<tolerance)and(convergance_type==1)):
            converged=True
            break
        previous_result=current_result
    if converged:
        if debug>0: print(f"Converged after {k+1} Iterations.")
    else:
        if debug>0: print("Did Not Converge.")
    if debug>0: print("")
    if debug>2: print("Number of Steps:",num_true_steps)
    if debug>2: print("")
    if debug>2: print("=================================================================================================================================================")
    if debug==2: print("=============================")
    if debug>1: print("")
    if(previous_settings==None):
        return(new_values,k+1)
    else:
        return(new_values,[max_gradient_step,con_step_down,con_no_step])

def gradient_descent_multi_sampled(initial_values,iterable,derivative,max_gradient_step,evaluator,tolerance,focal_width,focal_depth,max_num_iterations,debug):
    num_rows=initial_values.shape[0]
    num_cols=initial_values.shape[1]
    new_values=initial_values
    previous_result=evaluator(new_values)
    converged=False
    if debug>1: num_true_steps=0
    for k in range(max_num_iterations):
        gradient=derivative(new_values)*iterable
        if debug>2:
            x=np.linspace(0,max_gradient_step,1000)
            y=np.empty(x.shape)
            for idx,i in enumerate(x):
                test_matrix=new_values-gradient*i
                y[idx]=evaluator(test_matrix)

            plt.plot(x,y)
            plt.xscale("linear")
            plt.show()
        if debug>1: print("Iteration:",k+1)
        if debug>1: print("Gradient\n",gradient)
        if debug>1: print("Gradient Magnitude:",np.sqrt(np.sum(np.power(gradient,2))))
        if debug>1: print("")
        low=0
        high=max_gradient_step
        for l in range(focal_depth):
            test_step_sizes=np.linspace(low,high,focal_width)
            results=np.empty(test_step_sizes.shape)
            for idx,size in enumerate(test_step_sizes):
                test_matrix=new_values-gradient*size
                results[idx]=evaluator(test_matrix)
            idx_of_min=np.argmin(results)
            low=test_step_sizes[idx_of_min-1] if idx_of_min>0 else 0
            high=test_step_sizes[idx_of_min+1] if idx_of_min<results.size-1 else test_step_sizes[idx_of_min]
            if debug>1: print("Focal Strength:",l+1)
            if debug>1: print("Best Step Size:",test_step_sizes[idx_of_min])
            if debug>1: print("Surrounding Sizes:",low,high)
            if debug>1: print("Previous Result",previous_result)
            if debug>1: print("Step Result:",results[idx_of_min])
            if debug>1: print("")
        if(results[idx_of_min]<previous_result):
            new_values=new_values-gradient*test_step_sizes[idx_of_min]
            previous_result=results[idx_of_min]
            if debug>1: num_true_steps+=1
        else:
            print("Error, Not low Enough")
            return(-1)
        if debug>1: print("Final Position:\n",new_values)
        if debug>1: print("")
        if debug>1: print("-------------------------------------------------------------------------------------------------------------------------------------------------")
        if debug==1: print(k+1)
        if debug>0: print("")
        if(np.sqrt(np.sum(np.power(gradient,2)))<tolerance):
            converged=True
            break
    if converged:
        print(f"Converged after {k+1} Iterations.")
    else:
        print("Did Not Converge.")
    print("")
    if debug>1: print("Number of Steps:",num_true_steps)
    if debug>1: print("")
    if debug>1: print("=================================================================================================================================================")
    if debug==1: print("=============================")
    if debug>0: print("")
    return(new_values)

def Gauss_Newton_Algorithm(initial_values,energy_length,evaluator,jacobian_func,error_func,tolerance,max_num_iterations,debug):
    num_cols=initial_values.shape[1]
    new_values=initial_values
    previous_result=evaluator(new_values)
    converged=False
    for k in range(max_num_iterations):
        jac=jacobian_func(energy_length,new_values)
        delta_value=np.linalg.inv(jac.T@jac)@jac.T@error_func(new_values)
        delta_matrix=np.reshape(delta_value,new_values.shape)
        if debug: print(k+1)
        if debug: print(delta_matrix)
        if debug: print("-----------------------------")
        if debug: print("")
        new_values=new_values+delta_matrix
        if(np.sqrt(np.sum(np.power(delta_matrix,2)))<tolerance):
            converged=True
            break
    if converged:
        print(f"Converged after {k+1} Iterations.")
    else:
        print("Did Not Converge.")
    print("")
    if debug: print("=============================")
    if debug: print("")
    return(new_values)

def LMA_Type_1(initial_values,energy_length,iterables,jacobian_func,error_func,lambda_params,metric_minimum,tolerance,max_num_iterations,debug):
    num_iterables=np.sum(iterables)
    new_values=initial_values
    converged=False
    for k in range(max_num_iterations):
        jac=jacobian_func(energy_length,new_values)
        iterable_indexes=np.where(np.ndarray.flatten(iterables)==1)[0]
        jac=jac[:,iterable_indexes]
        
        delta_vals=np.linalg.inv(jac.T@jac+lambda_params[0]*np.eye(num_iterables))@jac.T@error_func(new_values)
        delta_matrix=np.zeros(new_values.size)
        delta_matrix[iterable_indexes]=delta_vals
        delta_matrix=np.reshape(delta_matrix,new_values.shape)
        
        if debug>0: print(k+1)
        if debug>1: print(delta_matrix)
        
        test_values=new_values+delta_matrix
        try:
            metric_num=np.sum(np.power(error_func(new_values),2))-np.sum(np.power(error_func(test_values),2))
            metric_denom=delta_vals.T@(lambda_params[0]*delta_vals+jac.T@error_func(new_values))
            metric=metric_num/metric_denom
        except:
            metric=0
        
        if debug>1:print(lambda_params[0])
        if debug>1: print(metric)
        if debug>1: print("-----------------------------")
        if debug>0: print("")
        
        if(metric>metric_minimum):
            new_values=test_values
            lambda_params[0]=np.max([lambda_params[0]/lambda_params[2],lambda_params[4]])
        else:
            lambda_params[0]=np.min([lambda_params[0]*lambda_params[1],lambda_params[3]])
        
        if(np.sqrt(np.sum(np.power(delta_matrix,2)))<tolerance):
            converged=True
            break
    
    if converged:
        print(f"Converged after {k+1} Iterations.")
    else:
        print("Did Not Converge.")
    print("")
    if debug>0: print("=============================")
    if debug>0: print("")
    return(new_values)

def LMA(initial_vector,
        evaluator,
        calculator,
        initial_priority,
        priority_multiplier,
        priority_min,
        priority_max,
        improvement_threshold,
        iteration_limit,
        debug,
        previous_settings=None):
    #Initialization
    v_length=len(initial_vector)
    priority=initial_priority
    vector=initial_vector
    evaluation=evaluator(vector)
    #LM Loop
    for iteration in range(iteration_limit):
        #Calculates Values
        gradient,hessian=calculator(vector)
        #Gets Step Size
        difference=np.linalg.inv(hessian+priority*np.eye(v_length))@gradient
        #Tests a step and evaluates it
        new_vector=vector-difference
        new_evaluation=evaluator(new_vector)
        error_change=evaluation-new_evaluation
        metric=error_change/np.abs(difference.T@(priority*difference+gradient))
        #Updates loop based on evaluation
        if(metric>improvement_threshold):
            priority=np.max([priority/priority_multiplier,priority_min])
            vector=new_vector
            evaluation=new_evaluation
        elif(priority):
            #Exits if no further improvement can be made in this format
            if(priority==priority_max):
                break
            priority=np.min([priority*priority_multiplier,priority_max])
    return(vector,iteration)

def modified_LMA(initial_vector,
                 evaluator,
                 calculator,
                 initial_priority,
                 priority_multiplier,
                 priority_min,
                 priority_max,
                 improvement_threshold,
                 iteration_limit,
                 debug,
                 previous_settings=None):
    #Initialization
    v_length=len(initial_vector)
    priority=initial_priority
    vector=initial_vector
    evaluation=evaluator(vector)
    #LM Loop
    for iteration in range(iteration_limit):
        #Calculates Values
        gradient,hessian=calculator(vector)
        #Gets Step Size
        difference=np.linalg.inv(hessian+priority*np.eye(v_length))@gradient
        #Tests a step and evaluates it
        new_vector=vector-difference
        new_evaluation=evaluator(new_vector)
        error_change=evaluation-new_evaluation
        metric=error_change/np.abs(difference.T@(priority*difference+gradient))
        #Updates loop based on evaluation
        if(metric>improvement_threshold):
            priority=np.max([priority/priority_multiplier,priority_min])
            vector=new_vector
            evaluation=new_evaluation
        elif(priority):
            #Exits if no further improvement can be made in this format
            if(priority==priority_max):
                break
            priority=np.min([priority*priority_multiplier,priority_max])
    return(vector,iteration)