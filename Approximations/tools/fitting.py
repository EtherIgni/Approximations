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

def gaussNewtonAlgorithm(initial_values,
                         energy_length,
                         evaluator,
                         jacobian_func,
                         error_func,
                         tolerance,
                         max_num_iterations,
                         debug):
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

def gradient_Descent(evaluation_model,
                    parameters,
                    debug):
    # Required Parameters:
    #   Iteration Limit
    #   Largest Gradient Step
    #   Max Depth Per Iteration
    #   Compounding Count Limit
    #   Convergence Tolerance
    #   Convergence Type
    #
    # Required Model Functionality:
    #   evaluate
    #   calcGradient

    evaluator  = evaluation_model.evaluate
    calculator = evaluation_model.calc_Gradient
    def fitting_Func(initial_vector,data):
        vector=initial_vector
        previous_result=evaluator(vector,data)
        converged=False
        if debug>2: num_true_steps=0
        con_step_down=0
        con_no_step=0
        for k in range(parameters["Iteration Limit"]):
            gradient=calculator(vector,data)
            if debug>3:
                x=np.linspace(0,parameters["Largest Gradient Step"],1000)
                y=np.empty(x.shape)
                for idx,i in enumerate(x):
                    test_vector=vector-gradient*i
                    y[idx]=evaluator(test_vector,data)

                plt.plot(x,y)
                plt.xscale("linear")
                plt.show()
            if debug>2: print("Iteration:",k+1)
            if debug>2: print("Gradient\n",gradient)
            if debug>2: print("Gradient Magnitude:",np.sqrt(np.sum(np.power(gradient,2))))
            current_step_size=parameters["Largest Gradient Step"]
            if debug>2: print("")
            for l in range(parameters["Max Depth Per Iteration"]):
                if debug>2: print("Step Count:",l+1)
                if debug>2: print("Step Size:",current_step_size)
                test_vector=vector-gradient*current_step_size
                if debug>2: print("New Position\n",test_vector)
                current_result=evaluator(test_vector,data)
                if debug>2: print("Result to Beat:",previous_result)
                if debug>2: print("Step Result:",current_result)
                if(current_result<previous_result):
                    vector=test_vector
                    if debug>2: num_true_steps+=1
                    if debug>2: print("Step")
                    if debug>2: print("")
                    break
                current_step_size=current_step_size/2
                if debug>2: print("")
            else:
                if debug>0: print("Failed to step Before Failsafe Triggered")
                break
            if debug>2: print("Final Position:\n",vector)
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
            if(con_no_step==parameters["Compounding Count Limit"]):
                max_gradient_step=max_gradient_step*2
                con_no_step=0
            if(con_step_down==parameters["Compounding Count Limit"]):
                max_gradient_step=max_gradient_step/2
                con_step_down=0
            if((np.sqrt(np.sum(np.power(gradient,2)))<parameters["Convergence Tolerance"])and(parameters["Convergence Type"]==0)):
                converged=True
                break
            if((current_result-previous_result<parameters["Convergence Tolerance"])and(parameters["Convergence Type"]==1)):
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
        return(vector,k+1)
    return(fitting_Func)

def levenberg_Marquardt(evaluation_model,
                       parameters,
                       debug):
    # Required Parameters:
    #   Iteration Limit
    #   Improvement Threshold
    #   Initial Priority
    #   Priority Multiplier
    #   Priority Minimum
    #   Priority Maximum
    #
    # Required Model Functionality:
    #   evaluate
    #   calcGradientAndHessian

    evaluator  = evaluation_model.evaluate
    calculator = evaluation_model.calc_Gradient_And_Hessian
    def fitting_Func(initial_vector,data):
        #Initialization
        v_length=len(initial_vector)
        priority=parameters["Initial Priority"]
        vector=initial_vector
        evaluation=evaluator(vector,data)
        #LM Loop
        for iteration in range(parameters["Iteration Limit"]):
            #Calculates Values
            gradient,hessian=calculator(vector,data)
            #Gets Step Size
            difference=np.linalg.inv(hessian+priority*np.eye(v_length))@gradient
            #Tests a step and evaluates it
            new_vector=vector-difference
            new_evaluation=evaluator(new_vector,data)
            error_change=evaluation-new_evaluation
            metric=error_change/np.abs(difference.T@(priority*difference+gradient))
            #Updates loop based on evaluation
            if(metric>parameters["Improvement Threshold"]):
                priority=np.max([priority/parameters["Priority Multiplier"],parameters["Priority Minimum"]])
                vector=new_vector
                evaluation=new_evaluation
            elif(priority):
                #Exits if no further improvement can be made in this format
                if(priority==parameters["Priority Maximum"]):
                    break
                priority=np.min([priority*parameters["Priority Multiplier"],parameters["Priority Maximum"]])
        return(vector,iteration)
    return(fitting_Func)