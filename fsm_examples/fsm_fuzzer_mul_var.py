# generates a counter with one unreachable state and 2 to 5 loops

import numpy as np

num_var = range(5, 500, 25)

num_states = 10

# guard_type = ["comb.icmp ult %", "comb.icmp uge %"]

action = "comb.add %"

for n in num_var:

    loops = []

    const = []


    output_file = "vars/fsm_"+str(n)+"var_10states.mlir"
    f = open(output_file, "w")


    vars = []
    rows = []
    cols = []
    guards = []
    actions = []
    guard_vals = []

    # initialize variables

    for v in range(n):
        vars.append("x"+str(v))

    for s in range(num_states):
        rows.append(s)
        cols.append(s+1)
        const = np.random.randint(1, 10)
        # standard transition, no guard, action only
        guards.append("NULL")
        tmp_a = []
        for v in range(n):
            tmp_a.append(action+"x"+str(n)+", %c"+str(const))
        actions.append(tmp_a)

    print("transitions: "+str(len(rows)))
    print("guards: "+str(len(guards)))
    print("actions: "+str(len(actions)))

    f.write("fsm.machine @fsm"+str(n)+"() -> () attributes {initialState = \"_0\"} {\n")
    for v in vars:
        f.write("\t%"+v+" = fsm.variable \""+v+"\" {initValue = 0 : i16} : i16\n")
    for c in range(10):
        f.write("\t%c"+str(c)+" = hw.constant "+str(c)+" : i16\n")

    for st in range(n):
        f.write("\n\n\tfsm.state @_"+str(st)+" output {\n\t} transitions {")
        for i in range(len(rows)):
            if rows[i] == st:
                f.write("\n\t\tfsm.transition @_"+str(cols[i]))
                if guards[i]!="NULL":
                    f.write("\n\t\t\tguard {")
                    f.write("\n\t\t\t\t%tmp = "+guards[i]+" : i16")
                    f.write("\n\t\t\t\tfsm.return %tmp")
                    f.write("\n\t\t\t} action {")
                else:
                    f.write("\n\t\taction {")
                    for tmp in actions[i]:
                        f.write("\n\t\t\t\t%tmp = "+tmp+" : i16")
                        f.write("\n\t\t\t\tfsm.update "+vars[actions[i].index(tmp)]+", %tmp : i16")
                f.write("\n\t\t\t}")
        f.write("\n\t}")
    
    # last state
    f.write("\n\n\tfsm.state @_"+str(n)+" output {\n\t} transitions {\n\t}")
        
    # non reachable state

    # f.write("\n\n\tfsm.state @_nr output {\n\t} transitions {")
    # f.write("\n\t\tfsm.transition @_0")
    # f.write("\n\t}")

    f.write("\n}")

    f.close()

    