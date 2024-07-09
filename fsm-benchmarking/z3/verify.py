import subprocess 
import os

reps = 10

folder = "../smtlib-fsm/err-p2/"

outputfile = open("z3-err-p2.txt", "w")

for file in os.listdir(folder):
    print(file)
    command = "z3 "+folder+file+" --st"
    print(command)
    for i in range(reps):
        outputfile.write("\nrep "+ str(i)+" file "+folder+file)
        process = subprocess.run([command], shell=True, capture_output=True, text=True)
        # Store the output
        output = process.stdout
        error_output = process.stderr
        # Print the outputs
        # print("Standard Output:")
        # print(output)
        # print("Error Output:")
        # print(error_output)
        outputfile.write(output)
        outputfile.write(error_output)


        
outputfile.close()
