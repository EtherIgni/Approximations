from subprocess import run
from time import sleep

# Path and name to the script you are trying to start
file_path = "Approximations/acquisition/data_generation.py" 

restart_timer = 2
def start_script():
    try:
        print("Running Script")
        run("python "+file_path, check=True)
    except:
        print("Scripted Crashed ,Restarting...")
        handle_crash()

def handle_crash():
    sleep(restart_timer)
    start_script()

start_script()