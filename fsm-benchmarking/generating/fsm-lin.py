# generates a counter with one unreachable state and 2 to 5 loops

import numpy as np

num_fsm = range(10, 500, 10)

guard_type = ["comb.icmp eq %", "comb.icmp ne %"]

action = "comb.add %"

for n in num_fsm:

    loops = []

    const = []

    nl = 0

    print("num loops: "+str(nl))

    while len(loops)<nl:
        tmp = np.random.randint(2, n)
        if tmp not in loops:
            loops.append(tmp)


    output_file = "../fsm/linear/fsm_"+str(n)+"states_"+str(nl)+"loops.mlir"
    f = open(output_file, "w")


    vars = []
    rows = []
    cols = []
    guards = []
    actions = []
    guard_vals = []

    # initialize variables

    var = "x0"

    for s in range(n):
        rows.append(s)
        cols.append(s+1)

        if(s in loops):
            g = np.random.randint(s, s*2)
            while g in guard_vals:
                g = np.random.randint(s, s*2)
            guard_vals.append(g)
            const.append(g)
            # transition with guard 
            guards.append(str(guard_type[1])+"x0, %c"+str(g))
            actions.append(action+"x0, %c1")
            backprop = np.random.randint(0, s-1)
            # also add loop transition
            rows.append(s)
            cols.append(backprop)
            guards.append(str(guard_type[0])+"x0, %c"+str(g))
            actions.append(action+"x0, %c1")
            print("from "+str(s)+" to "+str(backprop))


        else:
            # standard transition, no guard, action only
            guards.append("NULL")
            actions.append(action+"x0, %c1")

        print("from "+str(s)+" to "+str(s+1))

    print("transitions: "+str(len(rows)))
    print("guards: "+str(len(guards)))
    print("actions: "+str(len(actions)))

    f.write("fsm.machine @fsm"+str(n)+"() -> () attributes {initialState = \"_0\"} {\n")
    f.write("\t%x0 = fsm.variable \"x0\" {initValue = 0 : i16} : i16\n")
    for c in const:
        f.write("\t%c"+str(c)+" = hw.constant "+str(c)+" : i16\n")
    f.write("\t%c1 = hw.constant 1 : i16\n")

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
                f.write("\n\t\t\t\t%tmp = "+actions[i]+" : i16")
                f.write("\n\t\t\t\tfsm.update %x0, %tmp : i16")
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

    