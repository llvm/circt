import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



reps = 10

tb1000 = "times100.txt"
tb1000 = "times200.txt"
tb1000 = "times300.txt"
tb1000 = "times400.txt"
tb1000 = "times500.txt"

# tb300 = "output_

file100 = open(tb1000)



col = ["#f0f9e8",
"#bae4bc",
"#7bccc4",
"#43a2ca",
"#0868ac"]


plt.rc('font', family='Helvetica')

states = {}

data = {}
for b in [100, 200, 300, 400, 500]:
    tb = "times"+str(b)+".txt"
    file = open(tb)
    s = []
    d = []
    l = file.readline()
    while (l):
        num_states = int(l.split(", ")[0].split("_")[1].split("s")[0])
        tmp = []
        for i in range(10):
            tmp.append(float(file.readline()))
        s.append(num_states)
        d.append(np.mean(tmp))
        tmp.clear()
        l = file.readline()
    data[b] = d
    states[b] = s



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


if(True):# and s20==s30 and s30==s40 and s40==s50):
    df = pd.DataFrame({'states':states[100], 'rtl,tB=100':data[100], 'rtl,tB=200':data[200], 'rtl,tB=300':data[300], 'rtl,tB=400':data[400], 'rtl,tB=500':data[500]})#, 'tB=300':d30, 'tB=400':d40, 'tB=500':d50})
    df1=df.sort_values(by=['states'])
    # print(df1)
    plot = df1.plot(x='states', color=col,  figsize=(10, 5))
    plt.xlabel('#states')
    # plt.ylabel('time [s]')

    tb100 = "../reachability-init0-lin-cpos/output_t100_cpos_lin_init0.txt"
    tb200 = "../reachability-init0-lin-cpos/output_t200_cpos_lin_init0.txt"
    tb300 = "../reachability-init0-lin-cpos/output_t300_cpos_lin_init0.txt"
    tb400 = "../reachability-init0-lin-cpos/output_t400_cpos_lin_init0.txt"
    tb500 = "../reachability-init0-lin-cpos/output_t500_cpos_lin_init0.txt"


    file10 = open(tb100)
    file20 = open(tb200)
    file30 = open(tb300)
    file40 = open(tb400)
    file50 = open(tb500)


    # plt.rc('font', family='Helvetica')
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
            print(tmp)

    print(d10)
    

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

    s30 = []
    d30=[]

    lines = file30.readlines()

    for l in lines:
        i=i+1
        num_states = int(l.split(", ")[0].split("_")[1].split("s")[0])
        if (i%reps == 0):
            s30.append(num_states)
            d30.append(np.mean(tmp))
            tmp.clear()
        else:
            tmp.append(float(l.split(", ")[1]))

    s40 = []
    d40=[]

    lines = file40.readlines()

    for l in lines:
        i=i+1
        num_states = int(l.split(", ")[0].split("_")[1].split("s")[0])
        if (i%reps == 0):
            s40.append(num_states)
            d40.append(np.mean(tmp))
            tmp.clear()
        else:
            tmp.append(float(l.split(", ")[1]))
    s50 = []
    d50=[]

    lines = file50.readlines()
    for l in lines:
        i=i+1
        num_states = int(l.split(", ")[0].split("_")[1].split("s")[0])
        if (i%reps == 0):
            s50.append(num_states)
            d50.append(np.mean(tmp))
            tmp.clear()
        else:
            tmp.append(float(l.split(", ")[1]))





    if(s10 == s20 and s20==s30 and s30==s40 and s40==s50):
        df = pd.DataFrame({'states':s10, 'fsm,tB=100':d10, 'fsm,tB=200':d20, 'fsm,tB=300':d30, 'fsm,tB=400':d40, 'fsm,tB=500':d50})
        df1=df.sort_values(by=['states'])
        # print(df1)
        df1.plot(x='states', color=col, ax=plot, style="--")
        plt.xlabel('#states')
    #     plt.ylabel('time [s]')    
    #     # plt.show()  
    #     # df2 = df1.transpose()
    #     # print(df2.index[0])
    #     # df2.columns=df1['states']#['s=5','s=30','s=55','s=80','s=105','s=30','s=35','s=40','s=45']
    #     # df2 = df2.drop(df2.index[0])
    #     # print(df2)


    #     # # plot = df2.plot()
    #     # # Line styles
    #     # line_styles = ['-', '--', '-.', ':','-', '--', '-.']

    #     # # Plot each column with a different line style
    #     # for i, (column, style) in enumerate(zip(df2.columns, line_styles)):
    #     #     plt.plot(df2.index, df2[column], label=column, linestyle=style, color =col[i%len(col)])

    #     # plt.xlabel('time bound (tB)')
    #     # plt.ylabel('time [s]')
    #     # plt.title('Time scaling to verify the same FSM increasing the time bound tB')

    plt.legend(ncol=2)

    # plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    # plt.spines['bottom'].set_visible(False)
    # plt.spines['left'].set_visible(False)
        


    plt.savefig("init0-lin-cpos-rtl-fsm.png", dpi=300)
    # else:
    #     print("ERR")

    # ## singular plot

    # # states = []
    # # data = []

    # # lines = outputfile.readlines()

    # # i = 0

    # # tmp = []

    # # for l in lines:
    # #     i=i+1
    # #     num_states = int(l.split(", ")[0].split("_")[1].split("s")[0])
    # #     if (i%reps == 0):
    # #         states.append(num_states)
    # #         data.append(np.mean(tmp))
    # #         tmp.clear()
    # #     else:
    # #         tmp.append(float(l.split(", ")[1]))


    # # df = pd.DataFrame({'states':states, 'time':data})

    # # print(df)

    # # plot = df.plot(kind='scatter',title=filename.split(".")[0], x='states', y='time', color='#F6B092')

    # # plt.xlabel('#states')
    # # plt.ylabel('time [s]')

    # # plt.savefig(filename.split(".")[0]+".pdf")
