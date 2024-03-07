import numpy as np
import subprocess 

reps = 10

for fsm in range(5, 50, 5):
    file = "fsm"+str(fsm)+".mlir"
    command = "../build/bin/fsm-verification ../fsm_examples/"+file
    print(command)
    for i in range(reps):
        print("rep "+ str(i)+" file "+file)
        process = subprocess.run([command], shell=True)
    # o, e = process.communicate()
    # print(o)
    # print(e)
