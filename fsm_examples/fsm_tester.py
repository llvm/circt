import numpy as np
import subprocess 
import os

reps = 10

for file in os.listdir("structured/"):
    print(file)
    command = "../build/bin/fsm-verification ../fsm_examples/structured/"+file+" 40"
    print(command)
    for i in range(reps):
        print("rep "+ str(i)+" file "+file)
        process = subprocess.run([command], shell=True)


# for fsm in range(5, 50, 5):
#     file = "fsm"+str(fsm)+".mlir"
#     command = "../build/bin/fsm-verification ../fsm_examples/"+file+" 10"
#     print(command)
#     for i in range(reps):
#         print("rep "+ str(i)+" file "+file)
#         process = subprocess.run([command], shell=True)
#     # o, e = process.communicate()
#     # print(o)
#     # print(e)
