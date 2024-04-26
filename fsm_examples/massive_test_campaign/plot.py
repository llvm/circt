import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



reps = 10

col = [#"#f0f9e8",
"#bae4bc",
"#7bccc4",
"#43a2ca",
"#0868ac"]

lines = open("output.txt", "r").readlines()



plt.rc('font', family='Helvetica')

tmp = []

states = []

vars = []

time = []

# fsm_2var_2500states_1250loop,2.48513,0

for l in range(len(lines)):
    if len(tmp) == 10 and l > 0:
        vars.append(int(lines[l-1].split(",")[0].split("_")[1].split("v")[0]))
        states.append(int(lines[l-1].split(",")[0].split("_")[2].split("s")[0]))
        time.append(np.average(tmp))
        tmp.clear() 
        tmp.append(float(lines[l].split(",")[1]))
    else:
        tmp.append(float(lines[l].split(",")[1]))



df = pd.DataFrame({'states':states, 'num. var':vars, 'time':time})

# line_styles = ['-', '--', '-.', ':','-', '--', '-.']


df1=df.sort_values(by=['states', 'num. var'])



# plt.rc('font', family='Helvetica')

grouped = df1.groupby('num. var')
i=0
# Plotting each group
for name, group in grouped:
    print(name)
    print(group)
    plt.plot(group['states'], group['time'], label=f'num. var={name}', color=col[i])
    i=i+1

plt.xlabel('#states')
plt.ylabel('time [s]')




plt.legend(ncol = 2)


plt.savefig("reachability-massive-tests.png", dpi = 300)

