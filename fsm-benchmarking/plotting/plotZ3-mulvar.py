import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

colors = ["#f0f9e8",
"#bae4bc",
"#7bccc4",
"#43a2ca",
"#0868ac"]

file = open("../z3/z3-linear-p1-3007-v1.txt", "r")

reps = 10

stateNum = []
time = []
tmpTime = []

for l in file.readlines():
    if "rep 0" in l: 
        state = int(l.split(" ")[3].split("/")[3].split("_")[2].split("s")[0])
        print(state)
        stateNum.append(state)
    if "total-time" in l: 
        tmpTime.append(float(l.split(" ")[-1].split(")")[0]))
        if(len(tmpTime)==reps):
            time.append(np.average(tmpTime))
            tmpTime=[]

df = pd.DataFrame({'time':time, '#states':stateNum})

df1 = df.sort_values(by='#states')

file2 = open("../z3/z3-linear-p1-3007-v2.txt", "r")

reps = 10

stateNum = []
time = []
tmpTime = []

for l in file2.readlines():
    if "rep 0" in l: 
        state = int(l.split(" ")[3].split("/")[3].split("_")[2].split("s")[0])
        print(state)
        stateNum.append(state)
    if "total-time" in l: 
        tmpTime.append(float(l.split(" ")[-1].split(")")[0]))
        if(len(tmpTime)==reps):
            time.append(np.average(tmpTime))
            tmpTime=[]

dft = pd.DataFrame({'time':time, '#states':stateNum})

df2 = dft.sort_values(by='#states')

file3 = open("../z3/z3-linear-p1-3007-v4.txt", "r")

reps = 10

stateNum = []
time = []
tmpTime = []

for l in file3.readlines():
    if "rep 0" in l: 
        state = int(l.split(" ")[3].split("/")[3].split("_")[2].split("s")[0])
        print(state)
        stateNum.append(state)
    if "total-time" in l: 
        tmpTime.append(float(l.split(" ")[-1].split(")")[0]))
        if(len(tmpTime)==reps):
            time.append(np.average(tmpTime))
            tmpTime=[]

dftt = pd.DataFrame({'time':time, '#states':stateNum})

df3 = dftt.sort_values(by='#states')

file4 = open("../z3/z3-linear-p1-3007-v8.txt", "r")

reps = 10

stateNum = []
time = []
tmpTime = []

for l in file4.readlines():
    if "rep 0" in l: 
        state = int(l.split(" ")[3].split("/")[3].split("_")[2].split("s")[0])
        print(state)
        stateNum.append(state)
    if "total-time" in l: 
        tmpTime.append(float(l.split(" ")[-1].split(")")[0]))
        if(len(tmpTime)==reps):
            time.append(np.average(tmpTime))
            tmpTime=[]

dfttt = pd.DataFrame({'time':time, '#states':stateNum})

df4 = dfttt.sort_values(by='#states')

# print(df1)
# print(df2)
# print(df3)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each DataFrame
ax.plot(df1['#states'], df1['time'], color=colors[1], label='linear-v1')
ax.plot(df2['#states'], df2['time'], color=colors[2], label='linear-v2')
ax.plot(df3['#states'], df3['time'], color=colors[3], label='linear-v4')
ax.plot(df4['#states'], df4['time'], color=colors[4], label='linear-v8')

# ax.plot(df2['#states'], df2['time'], color=colors[2], label='1-loop')
# ax.plot(df3['#states'], df3['time'], color=colors[3], label='err-state')


# Set title and labels
ax.set_title('Verification time - scaling wrt. number of variables')
ax.set_xlabel('#states')
ax.set_ylabel('time')

# Add a legend
ax.legend()

# Show the plot
plt.savefig('plots/mulvar-z3.pdf')