import numpy as np
import subprocess 
import os

reps = 10

time = 100

for file in os.listdir("selfloop/"):
    to_check = int(file.split("_")[3].split(".")[0])
    states = int(file.split("_")[1].split("s")[0])
    time = to_check + 6
    command = "../build/bin/fsm-verification ../fsm_examples/selfloop/"+file+" "+str(time)+" "+str(to_check)+" "+str(to_check+3)
    print(command)
    if states <150:
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
