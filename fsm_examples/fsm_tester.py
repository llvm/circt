import numpy as np
import subprocess 
import os

reps = 10

time = 10000

for file in os.listdir("massive_test_campaign/testbench/"):
    print(file)
    loop = int(file.split("_")[3].split("l")[0])+5
    command = "../build/bin/fsm-verification ../fsm_examples/massive_test_campaign/testbench/"+file+" "+str(time)+" _"+str(loop)
    print(command)
    for i in range(reps):
        print("rep "+ str(i)+" file "+file)
        process = subprocess.run([command], shell=True)

