def strip(string):
    return(string.lstrip("= \n").rstrip("= \n"))

def split_parameter(string):
    string_split = string.split(":")
    
    name      = strip(string_split[0])
    data_type = strip(string_split[1]).split(",")
    value     = strip(string_split[2])
    
    if(data_type[0] == "s"):
        pass
    elif(data_type[0] == "i"):
        value = int(value)
    elif(data_type[0] == "f"):
        value = float(value)
    elif(data_type[0] == "l"):
        if(data_type[1] == "s"):
            value = value.split(",")
        elif(data_type[1] == "i"):
            value = [int(x) for x in value.split(",")]
        elif(data_type[1] == "f"):
            value = [float(x) for x in value.split(",")]
        else:
            raise Exception("Unsupported Data Type")
    else:
        raise Exception("Unsupported Data Type")
    
    return(name, value)

def input_Sheet_Translator(sheet_path):
    run_information         = {}
    molecular_information   = {}
    interaction_information = {}
    model_information       = {}
    fitting_parameters      = {}
    selections              = {}
    
    association_dict        = {"Run Meta Data":           run_information,
                               "Molecular Information":   molecular_information,
                               "Interaction Information": interaction_information,
                               "Model Information":       model_information,
                               "Fitting Parameters":      fitting_parameters,
                               "Selections":              selections}
    
    with open(sheet_path, "r") as file:
        lines = [line for line in file if (line!="\n")and(line[0]!="#")]
        
        header_indices = [idx for idx,line in enumerate(lines) if line[0:3]=="==="]
        header_indices.append(-1)
        
        for index in range(len(header_indices)-1):
            data_class = strip(lines[header_indices[index]][3:-4])
            parameters = lines[header_indices[index]+1:header_indices[index+1]]
            
            target_dict = association_dict[data_class]
            for parameter in parameters:
                name, value       = split_parameter(parameter)
                target_dict[name] = value
    
    return(run_information,
           molecular_information,
           interaction_information,
           model_information,
           fitting_parameters,
           selections)