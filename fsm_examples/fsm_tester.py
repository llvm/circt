import numpy as np
import subprocess 
import os

reps = 10

time = 20

for file in os.listdir("linear/"):
    to_check = int(file.split("_")[1].split("s")[0])-2
    if(to_check<150):
        # if to_check > time:
        #     to_check = time - 2
        print(file)
        command = "../build/bin/fsm-verification ../fsm_examples/linear/"+file+" "+str(time)+" "+str(time-1)
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
