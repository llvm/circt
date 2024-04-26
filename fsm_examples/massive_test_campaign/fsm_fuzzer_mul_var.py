# generates a counter with one unreachable state and 2 to 5 loops

import numpy as np

num_var = range(1, 5, 1)
num_states = range(50,5000,50)

guard_type = ["comb.icmp ult %", "comb.icmp uge %"]

action = "comb.add "

print(num_var)

print(num_states)


for n in num_var:

    for m in num_states:

        loop = np.floor(m/2)


        output_file = "testbench/fsm_"+str(n)+"var_"+str(m)+"states_"+str(int(loop))+"loop.mlir"

        f = open(output_file, "w")


        vars = []
        rows = []
        cols = []
        const = [1]
        guards = []
        actions = []
        guard_vals = []

        # initialize variables

        for v in range(n):
            vars.append("%x"+str(v))
    
        # print(vars)

        for s in range(m):
            rows.append(s)
            cols.append(s+1)
            if s == loop:
                g = np.random.randint(s, s*2)
                guard_vals.append(g)
                const.append(g)
                # transition with guard 
                guards.append(str(guard_type[1])+"x0, %c"+str(g))
                tmp_a = []
                for v in vars:
                    tmp_a.append(action+v+", %c1")
                actions.append(tmp_a)
                backprop = np.random.randint(0, s-1)
                # also add loop transition
                rows.append(s)
                cols.append(backprop)
                guards.append(str(guard_type[0])+"x0, %c"+str(g))
                tmp_b = []
                for v in vars:
                    tmp_a.append(action+v+", %c1")
                actions.append(tmp_b)
                print("from "+str(s)+" to "+str(backprop))

            else:
                # standard transition, no guard, action only
                guards.append("NULL")
                tmp_a = []
                for v in vars:
                    tmp_a.append(action+v+", %c1")
                actions.append(tmp_a)

        print("transitions: "+str(len(rows)))
        print("guards: "+str(len(guards)))
        print("actions: "+str(len(actions)))

        f.write("fsm.machine @fsm"+str(n)+"() -> () attributes {initialState = \"_0\"} {\n")
        for v in vars:
            f.write("\t"+v+" = fsm.variable \""+v+"\" {initValue = 0 : i16} : i16\n")
        for c in const:
            f.write("\t%c"+str(c)+" = hw.constant "+str(c)+" : i16\n")

        for st in range(m):
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
                        i = 0
                        for tmp in actions[i]:
                            f.write("\n\t\t\t\t%tmp"+str(i)+" = "+tmp+" : i16")
                            f.write("\n\t\t\t\tfsm.update "+vars[actions[i].index(tmp)]+", %tmp"+str(i)+" : i16")
                            i+=1
                    f.write("\n\t\t\t}")
            f.write("\n\t}")
        
        # last state
        f.write("\n\n\tfsm.state @_"+str(m)+" output {\n\t} transitions {\n\t}")
            
        # non reachable state

        # f.write("\n\n\tfsm.state @_nr output {\n\t} transitions {")
        # f.write("\n\t\tfsm.transition @_0")
        # f.write("\n\t}")

        f.write("\n}")

        f.close()

        