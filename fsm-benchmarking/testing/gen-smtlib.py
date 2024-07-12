import numpy as np
import subprocess 
import os

prop = 2

# example command
# ninja && valgrind bin/fsm-verification ../fsm_examples/input-err-state/errstate-fsm/fsm_5states_0loops.mlir ../fsm_examples/ltl_p3.mlir 4

folder ="fsm/linear/"

target="smtlib-fsm/linear-p2/"


for filename in os.listdir("../"+target):
    file_path = os.path.join("../"+target, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

for file in os.listdir("../"+folder):
    print(file)
    states = int(file.split("_")[1].split("s")[0])
    
    time = states+10

    ltl = open("tmp_prop.mlir", "w")

    if prop == 1:
        ltl.write("%state = unrealized_conversion_cast to !ltl.property\n")
        ltl.write("%e0 = ltl.eventually %state {state = \"_"+str(states-1)+"\"} : !ltl.property")

    elif prop == 2:
        ltl.write("%state = unrealized_conversion_cast to !ltl.property\n")
        ltl.write("%e0 = ltl.not %state {state = \"_"+str(states-1)+"\", var = \"0\", value = \""+str(states-1)+"\"} : !ltl.property")

    elif prop == 3:
        ltl.write("%error = unrealized_conversion_cast to !ltl.sequence\n")
        ltl.write("%state = unrealized_conversion_cast to !ltl.sequence\n")
        ltl.write("%e0 = ltl.implication %error, %state {state = \"ERR\", signal= \"0\", input = \"1\"}: !ltl.sequence, !ltl.sequence")

    ltl.close()

    command = "../../build/bin/fsm-verification ../../fsm-benchmarking/"+folder+file+" ../../fsm-benchmarking/testing/tmp_prop.mlir ../../fsm-benchmarking/"+target+file.split(".mlir")[0]+".smt "+str(time)
    print(command)

    process = subprocess.run([command], shell=True)
