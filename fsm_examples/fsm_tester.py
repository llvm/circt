import numpy as np
import subprocess 
import os

reps = 1

# ninja && valgrind bin/fsm-verification ../fsm_examples/input-err-state/errstate-fsm/fsm_5states_0loops.mlir ../fsm_examples/ltl_p3.mlir 4




for file in os.listdir("linear"):
    print(file)
    states = int(file.split("_")[1].split("s")[0])

    ltl = open("../fsm_examples/tmp_prop.mlir", "w")

    ltl.write("%state = unrealized_conversion_cast to !ltl.property\n")

    ltl.write("%e0 = ltl.eventually %state {state = \""+str(states-1)+"\"} : !ltl.property")

    ltl.close()

    command = "../build/bin/fsm-verification ../fsm_examples/linear/"+file+" ../fsm_examples/tmp_prop.mlir ../fsm_examples/linear-smtlib-p1/"+file.split(".mlir")[0]+".smt"
    print(command)
    for i in range(reps):
        print("rep "+ str(i)+" file "+file)
        process = subprocess.run([command], shell=True)

