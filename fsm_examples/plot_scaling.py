import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



reps = 10

tb100 = "output_t100_cnull_reachsat.txt"
tb200 = "output_t100_cnull_reachunsat.txt"
# tb300 = "output_t300_cpos_rp0neg.txt"
# tb400 = "output_t400_cpos_rp0neg.txt"
# tb500 = "output_t500_cpos_rp0neg.txt"


file10 = open(tb100)
file20 = open(tb200)
# file30 = open("Desktop/"+tb300)
# file40 = open("Desktop/"+tb400)
# file50 = open("Desktop/"+tb500)



col = ["#FB9F89", "#C4AF9A","#81AE9D","#2F323A"]

plt.rc('font', family='Helvetica')


i = 0

tmp = []

s10 = []
d10=[]

lines = file10.readlines()

for l in lines:
    i=i+1
    num_states = int(l.split(", ")[0].split("_")[1].split("s")[0])
    if (i%reps == 0):
        s10.append(num_states)
        d10.append(np.mean(tmp))
        tmp.clear()
    else:
        tmp.append(float(l.split(", ")[1]))

s20 = []
d20=[]

lines = file20.readlines()

for l in lines:
    i=i+1
    num_states = int(l.split(", ")[0].split("_")[1].split("s")[0])
    if (i%reps == 0):
        s20.append(num_states)
        d20.append(np.mean(tmp))
        tmp.clear()
    else:
        tmp.append(float(l.split(", ")[1]))

# s30 = []
# d30=[]

# lines = file30.readlines()

# for l in lines:
#     i=i+1
#     num_states = int(l.split(", ")[0].split("_")[1].split("s")[0])
#     if (i%reps == 0):
#         s30.append(num_states)
#         d30.append(np.mean(tmp))
#         tmp.clear()
#     else:
#         tmp.append(float(l.split(", ")[1]))

# s40 = []
# d40=[]

# lines = file40.readlines()

# for l in lines:
#     i=i+1
#     num_states = int(l.split(", ")[0].split("_")[1].split("s")[0])
#     if (i%reps == 0):
#         s40.append(num_states)
#         d40.append(np.mean(tmp))
#         tmp.clear()
#     else:
#         tmp.append(float(l.split(", ")[1]))
# s50 = []
# d50=[]

# lines = file50.readlines()
# for l in lines:
#     i=i+1
#     num_states = int(l.split(", ")[0].split("_")[1].split("s")[0])
#     if (i%reps == 0):
#         s50.append(num_states)
#         d50.append(np.mean(tmp))
#         tmp.clear()
#     else:
#         tmp.append(float(l.split(", ")[1]))




if(s10 == s20):# and s20==s30 and s30==s40 and s40==s50):
    df = pd.DataFrame({'states':s10, 'tB=100,S':d10, 'tB=100,U':d20})# 'tB=300':d30, 'tB=400':d40, 'tB=500':d50})
    df1=df.sort_values(by=['states'])
    # print(df1)
    plot = df1.plot(x='states', color=col, title='Time scaling for different-sized FSMs in relation to time bound (tB)')
    plt.xlabel('#states')
    plt.ylabel('time [s]')    
    # plt.show()  
    # df2 = df1.transpose()
    # print(df2.index[0])
    # df2.columns=df1['states']#['s=5','s=30','s=55','s=80','s=105','s=30','s=35','s=40','s=45']
    # df2 = df2.drop(df2.index[0])
    # print(df2)


    # # plot = df2.plot()
    # # Line styles
    # line_styles = ['-', '--', '-.', ':','-', '--', '-.']

    # # Plot each column with a different line style
    # for i, (column, style) in enumerate(zip(df2.columns, line_styles)):
    #     plt.plot(df2.index, df2[column], label=column, linestyle=style, color =col[i%len(col)])

    # plt.xlabel('time bound (tB)')
    # plt.ylabel('time [s]')
    # plt.title('Time scaling to verify the same FSM increasing the time bound tB')

    # plt.legend()
        


    plt.savefig("reach_sat_unsat.pdf")
else:
    print("ERR")

## singular plot

# states = []
# data = []

# lines = outputfile.readlines()

# i = 0

# tmp = []

# for l in lines:
#     i=i+1
#     num_states = int(l.split(", ")[0].split("_")[1].split("s")[0])
#     if (i%reps == 0):
#         states.append(num_states)
#         data.append(np.mean(tmp))
#         tmp.clear()
#     else:
#         tmp.append(float(l.split(", ")[1]))


# df = pd.DataFrame({'states':states, 'time':data})

# print(df)

# plot = df.plot(kind='scatter',title=filename.split(".")[0], x='states', y='time', color='#F6B092')

# plt.xlabel('#states')
# plt.ylabel('time [s]')

# plt.savefig(filename.split(".")[0]+".pdf")