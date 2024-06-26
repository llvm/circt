import numpy as np
import pandas as pd

filecomb = open("cvc5-lin-comb.txt", "r")

filep3 = open("cvc5-errstate-p3.txt", "r")

filep1 = open("cvc5-lin-p1.txt", "r")


reps = 10

current_filename = -1
filenames = []
times = []
current_times = []

for line in filecomb.readlines():
    if "rep" in line:
        current_filename = int(line.split(" ")[3].split("_")[1].split("s")[0])
    if "global::totalTime" in line:
        current_times.append(int(line.split(" ")[2].split("m")[0]))
        if len(current_times) == 10:
            times.append(np.average(current_times)/1000)
            filenames.append(current_filename)
            current_times.clear()
            current_filename = -1


df1 = pd.DataFrame({'states':filenames, 'exec-time':times})

current_filename = -1
filenames = []
times = []
current_times = []

for line in filep3.readlines():
    if "rep" in line:
        current_filename = int(line.split(" ")[3].split("_")[1].split("s")[0])
    if "global::totalTime" in line:
        current_times.append(int(line.split(" ")[2].split("m")[0]))
        if len(current_times) == 10:
            times.append(np.average(current_times)/1000)
            filenames.append(current_filename)
            current_times.clear()
            current_filename = -1


df2 = pd.DataFrame({'states':filenames, 'exec-time':times})

current_filename = -1
filenames = []
times = []
current_times = []

for line in filep1.readlines():
    if "rep" in line:
        current_filename = int(line.split(" ")[3].split("_")[1].split("s")[0])
    if "global::totalTime" in line:
        current_times.append(int(line.split(" ")[2].split("m")[0]))
        if len(current_times) == 10:
            times.append(np.average(current_times)/1000)
            filenames.append(current_filename)
            current_times.clear()
            current_filename = -1


df3 = pd.DataFrame({'states':filenames, 'exec-time':times})