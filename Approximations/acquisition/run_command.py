import os

from Approximations.acquisition.run_orders import run_order



sheets_path = os.getcwd()+"/Approximations/input_sheets"



orders = os.listdir(sheets_path+"/to_run")
if(len(orders)>0):
    for order in orders:
        print("=== Starting order "+order.split(".")[0]+" ===")
        run_order(sheets_path+"/to_run/"+order)
        print()
        print("=== Order finished ===")
        os.replace(sheets_path+"/to_run/"+order, sheets_path+"/previously_run/"+order)
        print()
    print("+++ All orders finished +++")
else:
    print("--- No orders to run ---")