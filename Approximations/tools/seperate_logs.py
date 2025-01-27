batch_path='/home/Aaron/Desktop/batch 3'

with open(batch_path+"/successful run data.txt","r") as file:
    lines=file.readlines()
    for line in lines:
        if(line[0]=='1'):
            with open(batch_path+"/srd split 1.txt","a") as file_new_1:
                file_new_1.write(line)
        else:
            with open(batch_path+"/srd split 2.txt","a") as file_new_2:
                file_new_2.write(line)

with open(batch_path+"/model data.txt","r") as file:
    lines=file.readlines()
    for line in lines:
        if(line[0]=='1'):
            with open(batch_path+"/md split 1.txt","a") as file_new_1:
                file_new_1.write(line)
        else:
            with open(batch_path+"/md split 2.txt","a") as file_new_2:
                file_new_2.write(line)
        