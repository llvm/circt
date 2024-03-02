import numpy as np

num_fsm = 10

for fsm in range(num_fsm):

    output_file = "fsm"+str(fsm)+".mlir"

    f = open(output_file, "w")

    num_states = 10

    num_trans = 15

    num_const = 10

    num_vars = 4

    vars = []
    rows = []
    cols = []
    guards = []
    action = []

    guard_type = ["comb.icmp eq ", "comb.icmp ne ", "comb.icmp slt ", "comb.icmp sle "]

    action_type = ["comb.add ", "comb.mul "]

    # initialize variables

    for i in range(num_vars):
        vars.append("x"+str(i))


    # first add enough transitions to make the FSM connected
    for s in range(num_states-1):
        print("from state ", s)
        print("to state ", s+1)
        rows.append(s)
        cols.append(s+1)
        guards.append(str(guard_type[np.random.randint(0, len(guard_type))])+str(np.random.randint(0, num_const)))
        print ("guard: ", guards[-1])
        action.append(str(action_type[np.random.randint(0, len(action_type))])+str(np.random.randint(0, num_const)))
        print ("action: ", action[-1])

    # then add more transitions to make the FSM interesting

    for s in range(num_trans-len(rows)):
        print("from state ", str(np.random.randint(0, num_states)))
        print("to state ", str(np.random.randint(0, num_states)))
        rows.append(s)
        cols.append(s+1)
        guards.append(str(guard_type[np.random.randint(0, len(guard_type))])+str(vars[np.random.randint(0, num_vars)])+", "+str(np.random.randint(0, num_const)))
        print ("guard: ", guards[-1])
        action.append(str(action_type[np.random.randint(0, len(action_type))])+str(vars[np.random.randint(0, num_vars)])+", "+str(np.random.randint(0, num_const)))
        print ("action: ", action[-1])

    # output proper mlir fsm

    f.write("fsm.machine @fsm"+str(fsm)+"() -> () {initialState = \"0\"} {\n")
    for v in vars:
        f.write("\t%"+v+" = fsm.variable \""+v+"\" {initValue = 0 : i16} : i16\n")
    for c in range(num_const):
        f.write("\t%c"+str(c)+" = hw.constant "+str(c)+" : i16\n")
    for st in range(num_states):
        f.write("\n\n\tfsm.state @"+str(st)+" output {\n\t} transitions {")
        for i in range(len(rows)):
            if rows[i] == st:
                f.write("\n\t\tfsm.transition @"+str(st+1)+" guard {")
                f.write("\n\t\t\t\t%tmp = "+guards[i]+" : i16")
                f.write("\n\t\t\t\tfsm.return %tmp")
                f.write("\n\t\t\t} action {")
                f.write("\n\t\t\t\t%tmp = "+action[i]+" : i16")
                f.write("\n\t\t\t\tfsm.update "+ str(vars[np.random.randint(0, num_vars)]) +", %tmp : i16")
                f.write("\n\t\t\t}")
        f.write("\n\t}")

    f.write("\n}")

    