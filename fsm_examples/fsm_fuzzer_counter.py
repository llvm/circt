import numpy as np

num_fsm = range(5, 100, 5)

for fsm in num_fsm:


    output_file = "fsm"+str(fsm)+".mlir"
    f = open(output_file, "w")
    


    num_states = fsm# np.random.randint(5, 20)

    num_trans = num_states

    num_const = 10

    num_vars = 4

    vars = []
    rows = []
    cols = []
    guards = []
    action = []

    guard_type = ["comb.icmp eq %", "comb.icmp ne %"]

    action_type = ["comb.add %", "comb.mul %"]

    # initialize variables

    for i in range(num_vars):
        vars.append("x"+str(i))


    # first add enough transitions to make the FSM connected
    for s in range(num_states-1):
        rows.append(s)
        cols.append(s+1)
        guards.append(str(guard_type[np.random.randint(0, len(guard_type))])+str(vars[np.random.randint(0, num_vars)])+", %c"+str(np.random.randint(0, num_const)))
        action.append(str(action_type[np.random.randint(0, len(action_type))])+str(vars[np.random.randint(0, num_vars)])+", %c"+str(np.random.randint(0, num_const)))


    # then add more transitions to make the FSM interesting
        
    print("row has ", len(rows))
    print("col has ", len(cols))

    for s in range(num_trans-len(rows)):
        tmpr = np.random.randint(0, num_states)
        tmpc = np.random.randint(0, num_states)
        unique = True 
        for i in range(len(rows)):
            if rows[i] == tmpr and cols[i] == tmpc:
                unique = False
        if(unique):
            rows.append(tmpr)
            cols.append(tmpc)
            guards.append(str(guard_type[np.random.randint(0, len(guard_type))])+str(vars[np.random.randint(0, num_vars)])+", %c"+str(np.random.randint(0, num_const)))
            action.append(str(action_type[np.random.randint(0, len(action_type))])+str(vars[np.random.randint(0, num_vars)])+", %c"+str(np.random.randint(0, num_const)))

    # output proper mlir fsm
            
    print("row has ", len(rows))
    print("col has ", len(cols))


    f.write("fsm.machine @fsm"+str(fsm)+"() -> () attributes {initialState = \"_0\"} {\n")
    for v in vars:
        f.write("\t%"+v+" = fsm.variable \""+v+"\" {initValue = 0 : i16} : i16\n")
    for c in range(num_const):
        f.write("\t%c"+str(c)+" = hw.constant "+str(c)+" : i16\n")
    for st in range(num_states):
        f.write("\n\n\tfsm.state @_"+str(st)+" output {\n\t} transitions {")
        for i in range(len(rows)):
            if rows[i] == st:
                f.write("\n\t\tfsm.transition @_"+str(cols[i])+" guard {")
                f.write("\n\t\t\t\t%tmp = "+guards[i]+" : i16")
                f.write("\n\t\t\t\tfsm.return %tmp")
                f.write("\n\t\t\t} action {")
                f.write("\n\t\t\t\t%tmp = "+action[i]+" : i16")
                f.write("\n\t\t\t\tfsm.update %"+ str(vars[np.random.randint(0, num_vars)]) +", %tmp : i16")
                f.write("\n\t\t\t}")
        f.write("\n\t}")

    f.write("\n}")

    f.close()

    