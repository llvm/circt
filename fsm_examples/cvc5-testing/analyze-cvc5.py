import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filecomb = open("cvc5-lin-p2.txt", "r")

filep3 = open("cvc5-lin-p3.txt", "r")

# filep1 = open("cvc5-lin-p1.txt", "r")
# 

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


df1 = pd.DataFrame({'states':filenames, 'exec-time-p2':times})

df1s = df1.sort_values(by=['states'])

print(df1s)

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


df2 = pd.DataFrame({'states':filenames, 'exec-time-p3':times})
df2s = df2.sort_values(by=['states'])


print(df2s)


# current_filename = -1
# filenames = []
# times = []
# current_times = []

# for line in filep1.readlines():
#     if "rep" in line:
#         current_filename = int(line.split(" ")[3].split("_")[1].split("s")[0])
#     if "global::totalTime" in line:
#         current_times.append(int(line.split(" ")[2].split("m")[0]))
#         if len(current_times) == 10:
#             times.append(np.average(current_times)/1000)
#             filenames.append(current_filename)
#             current_times.clear()
#             current_filename = -1


# df3 = pd.DataFrame({'states':filenames, 'exec-time':times})

fig, ax = plt.subplots(figsize=(10, 6))

col = [#"#f0f9e8",
"#bae4bc",
"#7bccc4",
"#43a2ca",
"#0868ac"]

ax.plot(df1s['states'], df1s['exec-time-p2'], color=col[0], label='comb-prop')
ax.plot(df2s['states'], df2s['exec-time-p3'], color=col[3], label='input-err-prop')

# Set title and labels
ax.set_title('cvc5 with different properties')
ax.set_xlabel('#states')
ax.set_ylabel('time')

# Add a legend
ax.legend()

# Show the plot


plt.savefig("tmp.png")
                