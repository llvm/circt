import numpy as np
import subprocess 
import os

reps = 10

time = 100

num_fsm = range(5, 50, 5)


for states in num_fsm:
    # print(file)
    # loop = int(file.split("_")[3].split("l")[0])+5
    command = "../../build/bin/fsm-verification ../../fsm_examples/massive-input-err-state/red_testbench/fsm_"+str(states)+"states_0loops.mlir "+str(time)+" ERR"
    print(command)
    for i in range(reps):
        print("rep "+ str(i)+" file "+str(states))
        process = subprocess.run([command], shell=True)

