import numpy as np
import subprocess 
import os

reps = 10

time = 100

for file in os.listdir("linear/"):
    print(file)
    command = "../build/bin/fsm-verification ../fsm_examples/linear/"+file+" "+str(time)+" _0"
    print(command)
    for i in range(reps):
        print("rep "+ str(i)+" file "+file)
        process = subprocess.run([command], shell=True)

